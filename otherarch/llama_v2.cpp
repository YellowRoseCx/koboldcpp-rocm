// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstdint>
#include <cstdio>
#endif

#include "llama_v2-util.h"
#include "llama_v2.h"

#include "ggml_v2.h"

#ifdef GGML_USE_CUDA
#include "ggml_v2-cuda.h"
#endif
#if defined(GGML_USE_CLBLAST)
#include "ggml_v2-opencl.h"
#endif


#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <numeric>

#define LLAMA_V2_USE_SCRATCH
#define LLAMA_V2_MAX_SCRATCH_BUFFERS 16

// available llama models
enum e_model2 {
    MODEL_UNKNOWN_2,
    MODEL_7B_2,
    MODEL_13B_2,
    MODEL_30B_2,
    MODEL_65B_2,
};

static const size_t MB_2 = 1024*1024;

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

static const std::map<e_model2, size_t> & MEM_REQ_SCRATCH0_2()
{
    static std::map<e_model2, size_t> k_sizes = {
        { MODEL_UNKNOWN_2, 512ull * MB_2 },
        { MODEL_7B_2,    512ull * MB_2 },
        { MODEL_13B_2,   512ull * MB_2 },
        { MODEL_30B_2,   640ull * MB_2 },
        { MODEL_65B_2,  1024ull * MB_2 },
    };
    return k_sizes;
}

static const std::map<e_model2, size_t> & MEM_REQ_SCRATCH1_2()
{
    static std::map<e_model2, size_t> k_sizes = {
        { MODEL_UNKNOWN_2, 512ull * MB_2 },
        { MODEL_7B_2,    512ull * MB_2 },
        { MODEL_13B_2,   512ull * MB_2 },
        { MODEL_30B_2,   640ull * MB_2 },
        { MODEL_65B_2,  1024ull * MB_2 },
    };
    return k_sizes;
}

// 2*n_embd*n_ctx*n_layer*sizeof(float16)
static const std::map<e_model2, size_t> & MEM_REQ_KV_SELF_2()
{
    static std::map<e_model2, size_t> k_sizes = {
        { MODEL_UNKNOWN_2, 1026ull * MB_2 },
        { MODEL_7B_2,   1026ull * MB_2 },
        { MODEL_13B_2,  1608ull * MB_2 },
        { MODEL_30B_2,  3124ull * MB_2 },
        { MODEL_65B_2,  5120ull * MB_2 },
    };
    return k_sizes;
}

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model2, size_t> & MEM_REQ_EVAL_2()
{
    static std::map<e_model2, size_t> k_sizes = {
        { MODEL_UNKNOWN_2,   800ull * MB_2 },
        { MODEL_7B_2,   800ull * MB_2 },
        { MODEL_13B_2, 1024ull * MB_2 },
        { MODEL_30B_2, 1280ull * MB_2 },
        { MODEL_65B_2, 1536ull * MB_2 },
    };
    return k_sizes;
}

// default hparams (LLaMA 7B)
struct llama_v2_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_mult  = 256;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;
    enum llama_v2_ftype ftype = LLAMA_V2_FTYPE_MOSTLY_F16;

    bool operator!=(const llama_v2_hparams & other) const {
        return memcmp(this, &other, sizeof(llama_v2_hparams));
    }
};

struct llama_v2_layer {
    // normalization
    struct ggml_v2_tensor * attention_norm;

    // attention
    struct ggml_v2_tensor * wq;
    struct ggml_v2_tensor * wk;
    struct ggml_v2_tensor * wv;
    struct ggml_v2_tensor * wo;

    // normalization
    struct ggml_v2_tensor * ffn_norm;

    // ff
    struct ggml_v2_tensor * w1;
    struct ggml_v2_tensor * w2;
    struct ggml_v2_tensor * w3;
};

struct llama_v2_kv_cache {
    struct ggml_v2_tensor * k;
    struct ggml_v2_tensor * v;

    struct ggml_v2_context * ctx = NULL;

    llama_v2_ctx_buffer buf;

    int n; // number of tokens currently in the cache

    ~llama_v2_kv_cache() {
        if (ctx) {
            ggml_v2_free(ctx);
        }
    }
};

struct llama_v2_model {
    e_model2 type = MODEL_UNKNOWN_2;

    llama_v2_hparams hparams;

    struct ggml_v2_tensor * tok_embeddings;

    struct ggml_v2_tensor * norm;
    struct ggml_v2_tensor * output;

    std::vector<llama_v2_layer> layers;

    // context
    struct ggml_v2_context * ctx = NULL;

    // key + value cache for the self attention
    // TODO: move to llama_v2_state
    struct llama_v2_kv_cache kv_self;

    // the model memory buffer
    llama_v2_ctx_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_v2_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_v2_mlock mlock_buf;
    llama_v2_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_v2_tensor *>> tensors_by_name;

    ~llama_v2_model() {
        if (ctx) {
            ggml_v2_free(ctx);
        }
    }
};

struct llama_v2_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct llama_v2_context {
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;
    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    llama_v2_model model;
    llama_v2_vocab vocab;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // memory buffers used to evaluate the model
    // TODO: move in llama_v2_state
    llama_v2_ctx_buffer buf_compute;
    llama_v2_ctx_buffer buf_scratch[LLAMA_V2_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_V2_MAX_SCRATCH_BUFFERS] = { 0 };

    void use_buf(struct ggml_v2_context * ctx, int i) {
#if defined(LLAMA_V2_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_v2_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_v2_set_scratch(ctx, { 0, buf.size, buf.addr, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_V2_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

template <typename T>
static T checked_mul2(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw format_old("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b);
    }
    return ret;
}

static size_t checked_div2(size_t a, size_t b) {
    if (b == 0 || a % b != 0) {
        throw format_old("error dividing %zu / %zu", a, b);
    }
    return a / b;
}

static std::string llama_v2_format_tensor_shape(const std::vector<uint32_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
    }
    return buf;
}

static size_t llama_v2_calc_tensor_size(const std::vector<uint32_t> & ne, enum ggml_v2_type type) {
    size_t size = ggml_v2_type_size(type);
    for (uint32_t dim : ne) {
        size = checked_mul2<size_t>(size, dim);
    }
    return size / ggml_v2_blck_size(type);
}

struct llama_v2_load_tensor_shard {
    std::vector<uint32_t> ne;
    size_t size;
    enum ggml_v2_type type;
    size_t file_idx;
    size_t file_off;

    void calc_size() {
        size = llama_v2_calc_tensor_size(ne, type);
    }
};

enum llama_v2_split_type {
    SPLIT_NONE_2,
    SPLIT_BY_COLUMNS_2,
    SPLIT_BY_ROWS_2
};

struct llama_v2_load_tensor {
    std::vector<llama_v2_load_tensor_shard> shards;

    std::string name;
    enum ggml_v2_type type = GGML_V2_TYPE_F32;
    llama_v2_split_type split_type = SPLIT_NONE_2;
    std::vector<uint32_t> ne;
    size_t size;
    struct ggml_v2_tensor * ggml_v2_tensor = NULL;
    uint8_t * data;

    llama_v2_load_tensor(const std::string & name) : name(name) {}

    void calc_all() {
        calc_type();
        calc_split_type();
        calc_ne();
        calc_size();
    }

    void calc_type() {
        const auto & first_shard = shards.at(0);
        for (const auto & shard : shards) {
            if (shard.type != first_shard.type) {
                throw format_old("inconsistent tensor shard type in '%s'", name.c_str());
            }
        }
        type = first_shard.type;
    }

    void calc_split_type() {
        if (shards.at(0).ne.size() == 1 || // 1D tensors are just duplicated in every file
            shards.size() == 1) { // only one file?
            split_type = SPLIT_NONE_2;
        } else if (name.find("tok_embeddings.") == 0 ||
            name.find(".attention.wo.weight") != std::string::npos ||
            name.find(".feed_forward.w2.weight") != std::string::npos) {
            split_type = SPLIT_BY_COLUMNS_2;
        } else {
            split_type = SPLIT_BY_ROWS_2;
        }
    }

    void calc_ne() {
        const auto & first_shard = shards.at(0);
        for (const auto & shard : shards) {
            if (shard.ne != first_shard.ne) {
                throw format_old("inconsistent tensor shard shape in '%s': first was %s, other was %s",
                             name.c_str(), llama_v2_format_tensor_shape(first_shard.ne).c_str(), llama_v2_format_tensor_shape(shard.ne).c_str());
            }
        }
        ne = first_shard.ne;
        LLAMA_V2_ASSERT(shards.size() <= UINT32_MAX);
        uint32_t n_shards = (uint32_t) shards.size();
        switch (split_type) {
            case SPLIT_NONE_2:
                ne = first_shard.ne;
                break;
            case SPLIT_BY_COLUMNS_2:
                ne = {checked_mul2<uint32_t>(first_shard.ne[0], n_shards),
                      first_shard.ne[1]};
                break;
            case SPLIT_BY_ROWS_2:
                ne = {first_shard.ne[0],
                      checked_mul2<uint32_t>(first_shard.ne[1], n_shards)};
                break;
        }
    }

    void calc_size() {
        size = llama_v2_calc_tensor_size(ne, type);
    }
};

struct llama_v2_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<llama_v2_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

enum llama_v2_file_version {
    LLAMA_V2_FILE_VERSION_GGML,
    LLAMA_V2_FILE_VERSION_GGMF_V1, // added version field and scores in vocab
    LLAMA_V2_FILE_VERSION_GGJT_V1, // added padding
    LLAMA_V2_FILE_VERSION_GGJT_V2, // changed quantization format
    LLAMA_V2_FILE_VERSION_GGJT_V3, // changed Q4 and Q8 quantization format
};

struct llama_v2_file_loader {
    llama_v2_file file;
    llama_v2_file_version file_version;
    llama_v2_hparams hparams;
    llama_v2_vocab vocab;

    llama_v2_file_loader(const char * fname, size_t file_idx, llama_v2_load_tensors_map & tensors_map)
        : file(fname, "rb") {
        fprintf(stderr, "llama.cpp: loading model from %s\n", fname);
        read_magic();
        read_hparams();
        read_vocab();
        read_tensor_metadata(file_idx, tensors_map);
    }
    void read_magic() {
        uint32_t magic = file.read_u32();
        uint32_t version = 0;

        uint32_t magic_ggjt = 0x67676a74u; // 'ggjt'
        uint32_t magic_ggmf = 0x67676d66u; // 'ggmf'
        uint32_t magic_ggml = 0x67676d6cu; // 'ggml'

        if (magic != magic_ggml) {
            version = file.read_u32();
        }

        if (magic == magic_ggml && version == 0) {
            file_version = LLAMA_V2_FILE_VERSION_GGML;
        } else if (magic == magic_ggmf && version == 1) {
            file_version = LLAMA_V2_FILE_VERSION_GGMF_V1;
        } else if (magic == magic_ggjt && version == 1) {
            file_version = LLAMA_V2_FILE_VERSION_GGJT_V1;
        } else if (magic == magic_ggjt && version == 2) {
            file_version = LLAMA_V2_FILE_VERSION_GGJT_V2;
        } else if (magic == magic_ggjt && version == 3) {
            file_version = LLAMA_V2_FILE_VERSION_GGJT_V3;
        } else {
            throw format_old("unknown (magic, version) combination: %08x, %08x; is this really a GGML file?",
                         magic, version);
        }
    }
    void read_hparams() {
        hparams.n_vocab = file.read_u32();
        hparams.n_embd = file.read_u32();
        hparams.n_mult = file.read_u32();
        hparams.n_head = file.read_u32();
        hparams.n_layer = file.read_u32();
        hparams.n_rot = file.read_u32();
        hparams.ftype = (enum llama_v2_ftype) file.read_u32();
    }
    void read_vocab() {
        vocab.id_to_token.resize(hparams.n_vocab);

        int32_t vocabloops = hparams.n_vocab;
        if(vocabloops==32001 && file_version == LLAMA_V2_FILE_VERSION_GGML)
        {
            printf("---\n!! WARNING: Model appears to be GPT4ALL v1 model, triggering compatibility fix !!\n---\n");
            vocabloops -= 1;
        }

        for (uint32_t i = 0; i < vocabloops; i++) {
            uint32_t len = file.read_u32();
            std::string word = file.read_string(len);

            float score = 0.0f;
            if (file_version >= LLAMA_V2_FILE_VERSION_GGMF_V1) {
                file.read_raw(&score, sizeof(score));
            }

            vocab.token_to_id[word] = i;

            auto & tok_score = vocab.id_to_token[i];
            tok_score.tok = std::move(word);
            tok_score.score = score;
        }
    }
    void read_tensor_metadata(size_t file_idx, llama_v2_load_tensors_map & tensors_map) {
        while (file.tell() < file.size) {
            llama_v2_load_tensor_shard shard;
            uint32_t n_dims = file.read_u32();
            uint32_t name_len = file.read_u32();
            shard.type = (enum ggml_v2_type) file.read_u32();
            shard.ne.resize(n_dims);
            file.read_raw(shard.ne.data(), sizeof(shard.ne[0]) * n_dims);
            std::string name = file.read_string(name_len);
            if (n_dims < 1 || n_dims > 2) {
                throw format_old("llama.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
            }
            switch (shard.type) {
                case GGML_V2_TYPE_F32:
                case GGML_V2_TYPE_F16:
                case GGML_V2_TYPE_Q4_0:
                case GGML_V2_TYPE_Q4_1:
                case GGML_V2_TYPE_Q4_2:
                case GGML_V2_TYPE_Q4_3:
                case GGML_V2_TYPE_Q5_0:
                case GGML_V2_TYPE_Q5_1:
                case GGML_V2_TYPE_Q8_0:
                    break;
                default: {
                    throw format_old("unrecognized tensor type %u\n", shard.type);
                }
            }

            if (file_version >= LLAMA_V2_FILE_VERSION_GGJT_V1) {
                // skip to the next multiple of 32 bytes
                file.seek(-file.tell() & 31, SEEK_CUR);
            }
            shard.file_idx = file_idx;
            shard.file_off = file.tell();

            shard.calc_size();
            file.seek(shard.size, SEEK_CUR);

            auto it = tensors_map.name_to_idx.find(name);
            size_t idx;
            if (it != tensors_map.name_to_idx.end()) {
                idx = it->second;
            } else {
                tensors_map.tensors.emplace_back(name);
                idx = tensors_map.tensors.size() - 1;
                tensors_map.name_to_idx.emplace(name, idx);
            }
            tensors_map.tensors.at(idx).shards.push_back(shard);
        }
    }
};

struct llama_v2_file_saver {
    llama_v2_file file;
    llama_v2_file_loader * any_file_loader;
    llama_v2_file_saver(const char * fname, llama_v2_file_loader * any_file_loader, enum llama_v2_ftype new_ftype)
        : file(fname, "wb"), any_file_loader(any_file_loader) {
        fprintf(stderr, "llama.cpp: saving model to %s\n", fname);
        write_magic();
        write_hparams(new_ftype);
        write_vocab();
    }
    void write_magic() {
        uint32_t magic_ggjt = 0x67676a74u; // 'ggjt'
        file.write_u32(magic_ggjt);   // magic
        file.write_u32(LLAMA_V2_FILE_VERSION); // version
    }
    void write_hparams(enum llama_v2_ftype new_ftype) {
        const llama_v2_hparams & hparams = any_file_loader->hparams;
        file.write_u32(hparams.n_vocab);
        file.write_u32(hparams.n_embd);
        file.write_u32(hparams.n_mult);
        file.write_u32(hparams.n_head);
        file.write_u32(hparams.n_layer);
        file.write_u32(hparams.n_rot);
        file.write_u32(new_ftype);
    }
    void write_vocab() {
        if (any_file_loader->file_version == LLAMA_V2_FILE_VERSION_GGML) {
            fprintf(stderr, "llama.cpp: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
        }
        uint32_t n_vocab = any_file_loader->hparams.n_vocab;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const auto & token_score = any_file_loader->vocab.id_to_token.at(i);
            file.write_u32((uint32_t) token_score.tok.size());
            file.write_raw(token_score.tok.data(), token_score.tok.size());
            file.write_raw(&token_score.score, sizeof(token_score.score));
        }
    }
    void write_tensor(llama_v2_load_tensor & tensor, enum ggml_v2_type new_type, const void * new_data, size_t new_size) {
        switch (new_type) {
            case GGML_V2_TYPE_F32:
            case GGML_V2_TYPE_F16:
            case GGML_V2_TYPE_Q4_0:
            case GGML_V2_TYPE_Q4_1:
            case GGML_V2_TYPE_Q4_2:
            case GGML_V2_TYPE_Q4_3:
            case GGML_V2_TYPE_Q5_0:
            case GGML_V2_TYPE_Q5_1:
            case GGML_V2_TYPE_Q8_0:
                break;
            default: LLAMA_V2_ASSERT(false);
        }
        file.write_u32((uint32_t) tensor.ne.size());
        file.write_u32((uint32_t) tensor.name.size());
        file.write_u32(new_type);
        file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
        file.write_raw(tensor.name.data(), tensor.name.size());
        file.seek(-file.tell() & 31, SEEK_CUR);
        LLAMA_V2_ASSERT(new_size == llama_v2_calc_tensor_size(tensor.ne, new_type));
        file.write_raw(new_data, new_size);
    }
};

struct llama_v2_model_loader {
    std::vector<std::unique_ptr<llama_v2_file_loader>> file_loaders;
    llama_v2_load_tensors_map tensors_map;
    bool use_mmap;
    size_t num_ggml_v2_tensors_created = 0;
    struct ggml_v2_context * ggml_v2_ctx = NULL;
    std::unique_ptr<llama_v2_mmap> mapping;

    llama_v2_model_loader(const std::string & fname_base, bool use_mmap, bool vocab_only) {
        auto * first_file = new llama_v2_file_loader(fname_base.c_str(), 0, tensors_map);
        file_loaders.emplace_back(first_file);
        uint32_t n_parts = vocab_only ? 1 : guess_n_parts();
        for (uint32_t i = 1; i < n_parts; i++) {
            std::string fname = fname_base + "." + std::to_string(i);
            auto * ith_file = new llama_v2_file_loader(fname.c_str(), i, tensors_map);
            file_loaders.emplace_back(ith_file);
            if (ith_file->hparams != first_file->hparams) {
                throw format_old("llama.cpp: hparams inconsistent between files");
            }
        }
        if (!llama_v2_mmap::SUPPORTED) {
            use_mmap = false;
        }
        if (use_mmap && alignment_prevents_mmap()) {
            fprintf(stderr, "llama.cpp: can't use mmap because tensors are not aligned; convert to new format to avoid this\n");
            use_mmap = false;
        }
        this->use_mmap = use_mmap;
        for (llama_v2_load_tensor & lt : tensors_map.tensors) {
            lt.calc_all();
        }
    }

    bool alignment_prevents_mmap() {
        for (const llama_v2_load_tensor & lt : tensors_map.tensors) {
            for (const llama_v2_load_tensor_shard & shard : lt.shards) {
                if (shard.file_off & 3) {
                    return true;
                }
            }
        }
        return false;
    }

    uint32_t guess_n_parts() const {
        auto it = tensors_map.name_to_idx.find("tok_embeddings.weight");
        if (it == tensors_map.name_to_idx.end()) {
            throw std::string("missing tok_embeddings.weight");
        }
        const llama_v2_load_tensor & lt = tensors_map.tensors.at(it->second);
        return file_loaders.at(0)->hparams.n_embd / lt.shards.at(0).ne.at(0);
    }

    void calc_sizes(size_t * ctx_size_p, size_t * mmapped_size_p) const {
        *ctx_size_p = *mmapped_size_p = 0;
        for (const llama_v2_load_tensor & lt : tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_v2_tensor) + GGML_V2_OBJECT_SIZE;
            *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size;
        }
    }

    struct ggml_v2_tensor * get_tensor(const std::string & name, const std::vector<uint32_t> & ne) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw format_old("llama.cpp: tensor '%s' is missing from model", name.c_str());
        }
        llama_v2_load_tensor & lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw format_old("llama.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                         name.c_str(), llama_v2_format_tensor_shape(ne).c_str(), llama_v2_format_tensor_shape(lt.ne).c_str());
        }

        return get_tensor_for(lt);
    }

    struct ggml_v2_tensor * get_tensor_for(llama_v2_load_tensor & lt) {
        struct ggml_v2_tensor * tensor;
        if (lt.ne.size() == 2) {
            tensor = ggml_v2_new_tensor_2d(ggml_v2_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            LLAMA_V2_ASSERT(lt.ne.size() == 1);
            tensor = ggml_v2_new_tensor_1d(ggml_v2_ctx, lt.type, lt.ne.at(0));
        }
        ggml_v2_set_name(tensor, lt.name.c_str());
        LLAMA_V2_ASSERT(lt.ggml_v2_tensor == NULL); // if this fails, we called get_tensor twice on the same tensor
        lt.ggml_v2_tensor = tensor;
        num_ggml_v2_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() const {
        if (num_ggml_v2_tensors_created != tensors_map.tensors.size()) {
            throw std::string("llama.cpp: file contained more tensors than expected");
        }
    }

    void load_all_data(llama_v2_progress_callback progress_callback, void *  progress_callback_user_data, llama_v2_mlock * lmlock) {
        size_t data_size = 0;
        for (const llama_v2_load_tensor & lt : tensors_map.tensors) {
            data_size += lt.size;
        }

        if (use_mmap) {
            mapping.reset(new llama_v2_mmap(&file_loaders.at(0)->file));
            if (!lmlock) {
                // Don't call the callback since the actual loading will be lazy
                // and we can't measure it.
                progress_callback = NULL;
            }
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (llama_v2_load_tensor & lt : tensors_map.tensors) {
            if (progress_callback) {
                progress_callback((float) done_size / data_size, progress_callback_user_data);
            }
            LLAMA_V2_ASSERT(lt.ggml_v2_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) lt.ggml_v2_tensor->data;
            load_data_for(lt);
            lt.ggml_v2_tensor->data = lt.data;
            done_size += lt.size;
            if (use_mmap && lmlock) {
                lmlock->grow_to(done_size);
            }
        }
        if (progress_callback) {
            progress_callback(1.0f, progress_callback_user_data);
        }
    }

    void load_data_for(llama_v2_load_tensor & lt) {
        if (use_mmap) {
            LLAMA_V2_ASSERT(lt.shards.size() == 1);
            lt.data = (uint8_t *) mapping->addr + lt.shards.at(0).file_off;
        } else if (lt.split_type == SPLIT_NONE_2) {
            llama_v2_file & file = file_loaders.at(lt.shards.at(0).file_idx)->file;
            file.seek(lt.shards.at(0).file_off, SEEK_SET);
            file.read_raw(lt.data, lt.size);
        } else if (lt.split_type == SPLIT_BY_ROWS_2) {
            size_t offset = 0;
            for (llama_v2_load_tensor_shard & shard : lt.shards) {
                llama_v2_file & file = file_loaders.at(shard.file_idx)->file;
                file.seek(shard.file_off, SEEK_SET);
                file.read_raw(lt.data + offset, shard.size);
                offset += shard.size;
            }
            LLAMA_V2_ASSERT(offset == lt.size);
        } else if (lt.split_type == SPLIT_BY_COLUMNS_2) {
            // Let's load the data into temporary buffers to ensure the OS performs large loads.
            std::vector<llama_v2_buffer> tmp_bufs(lt.shards.size());
            for (size_t i = 0; i < lt.shards.size(); i++) {
                llama_v2_load_tensor_shard & shard = lt.shards.at(i);
                llama_v2_file & file = file_loaders.at(shard.file_idx)->file;
                file.seek(shard.file_off, SEEK_SET);
                tmp_bufs.at(i).resize(shard.size);
                file.read_raw(tmp_bufs.at(i).addr, shard.size);
            }
            // Then reshape.
            size_t num_rows = lt.ne.at(1);
            size_t per_shard_row_size = lt.shards.at(0).size / num_rows;
            size_t out_offset = 0;
            for (size_t row = 0; row < num_rows; row++) {
                for (llama_v2_buffer & tmp_buf : tmp_bufs) {
                    memcpy(lt.data + out_offset,
                           tmp_buf.addr + row * per_shard_row_size,
                           per_shard_row_size);
                    out_offset += per_shard_row_size;
                }
            }
            LLAMA_V2_ASSERT(out_offset == lt.size);
        }
        if (0) {
            print_checksum(lt);
        }
    }

    static void print_checksum(llama_v2_load_tensor & lt) {
        uint32_t sum = 0;
        for (size_t i = 0; i < lt.size; i++) {
            uint8_t byte = lt.data[i];
            sum = byte + (sum << 6) + (sum << 16) - sum; // sdbm hash
        }
        fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
                llama_v2_format_tensor_shape(lt.ne).c_str(), lt.size);
    }

};


//
// kv cache
//

static bool kv_cache_init(
        const struct llama_v2_hparams & hparams,
             struct llama_v2_kv_cache & cache,
                         ggml_v2_type   wtype,
                               int   n_ctx) {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_v2_type_size(wtype) + 2u*MB_2);

    struct ggml_v2_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.addr;
    params.no_alloc   = false;

    cache.ctx = ggml_v2_init(params);

    if (!cache.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_v2_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_v2_new_tensor_1d(cache.ctx, wtype, n_elements);
    ggml_v2_set_name(cache.k, "cache_k");
    ggml_v2_set_name(cache.v, "cache_v");

    return true;
}

struct llama_v2_context_params llama_v2_context_default_params() {
    struct llama_v2_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.gpu_layers                  =*/ 0,
        /*.seed                        =*/ -1,
        /*.f16_kv                      =*/ true,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
    };

    return result;
}

bool llama_v2_mmap_supported() {
    return llama_v2_mmap::SUPPORTED;
}

bool llama_v2_mlock_supported() {
    return llama_v2_mlock::SUPPORTED;
}

//
// model loading
//

static const char *llama_v2_file_version_name(llama_v2_file_version version) {
    switch (version) {
        case LLAMA_V2_FILE_VERSION_GGML: return "'ggml' (old version with low tokenizer quality and no mmap support)";
        case LLAMA_V2_FILE_VERSION_GGMF_V1: return "ggmf v1 (old version with no mmap support)";
        case LLAMA_V2_FILE_VERSION_GGJT_V1: return "ggjt v1 (pre #1405)";
        case LLAMA_V2_FILE_VERSION_GGJT_V2: return "ggjt v2 (pre #1508)";
        case LLAMA_V2_FILE_VERSION_GGJT_V3: return "ggjt v3 (latest)";
    }

    return "unknown";
}

static const char *llama_v2_ftype_name(enum llama_v2_ftype ftype) {
    switch (ftype) {
        case LLAMA_V2_FTYPE_ALL_F32:     return "all F32";
        case LLAMA_V2_FTYPE_MOSTLY_F16:  return "mostly F16";
        case LLAMA_V2_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case LLAMA_V2_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case LLAMA_V2_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case LLAMA_V2_FTYPE_MOSTLY_Q4_2: return "mostly Q4_2";
        case LLAMA_V2_FTYPE_MOSTLY_Q4_3: return "mostly Q4_3";
        case LLAMA_V2_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case LLAMA_V2_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case LLAMA_V2_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";
        default:                      return "unknown, may not work";
    }
}

static const char *llama_v2_model_type_name(e_model2 type) {
    switch (type) {
        case MODEL_7B_2: return "7B";
        case MODEL_13B_2: return "13B";
        case MODEL_30B_2: return "30B";
        case MODEL_65B_2: return "65B";
        default:
            printf("\nWARNING: NON-STANDARD LLAMA FILE DETECTED. DEFAULT TO 7B SIZE.\n");
            return "UNKNOWN";
    }
}

static void llama_v2_model_load_internal(
        const std::string & fname,
        llama_v2_context & lctx,
        int n_ctx,
        int n_gpu_layers,
        ggml_v2_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_v2_progress_callback progress_callback,
        void * progress_callback_user_data) {

    lctx.t_start_us = ggml_v2_time_us();

    std::unique_ptr<llama_v2_model_loader> ml(new llama_v2_model_loader(fname, use_mmap, vocab_only));

    lctx.vocab = std::move(ml->file_loaders.at(0)->vocab);
    auto & model = lctx.model;
    model.hparams = ml->file_loaders.at(0)->hparams;
    llama_v2_file_version file_version = ml->file_loaders.at(0)->file_version;
    auto & hparams = model.hparams;
    uint32_t n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

    {
        switch (hparams.n_layer) {
            case 32: model.type = e_model2::MODEL_7B_2; break;
            case 40: model.type = e_model2::MODEL_13B_2; break;
            case 60: model.type = e_model2::MODEL_30B_2; break;
            case 80: model.type = e_model2::MODEL_65B_2; break;
            default: model.type = e_model2::MODEL_UNKNOWN_2; break;
        }

        hparams.n_ctx = n_ctx;
    }

    {
        fprintf(stderr, "%s: format     = %s\n",  __func__, llama_v2_file_version_name(file_version));
        fprintf(stderr, "%s: n_vocab    = %u\n",  __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_ctx      = %u\n",  __func__, hparams.n_ctx);
        fprintf(stderr, "%s: n_embd     = %u\n",  __func__, hparams.n_embd);
        fprintf(stderr, "%s: n_mult     = %u\n",  __func__, hparams.n_mult);
        fprintf(stderr, "%s: n_head     = %u\n",  __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer    = %u\n",  __func__, hparams.n_layer);
        fprintf(stderr, "%s: n_rot      = %u\n",  __func__, hparams.n_rot);
        fprintf(stderr, "%s: ftype      = %u (%s)\n", __func__, hparams.ftype, llama_v2_ftype_name(hparams.ftype));
        fprintf(stderr, "%s: n_ff       = %u\n",  __func__, n_ff);
        fprintf(stderr, "%s: n_parts    = %zu\n", __func__, ml->file_loaders.size());
        fprintf(stderr, "%s: model size = %s\n",  __func__, llama_v2_model_type_name(model.type));
    }

    if (file_version < LLAMA_V2_FILE_VERSION_GGJT_V2) {
        if (hparams.ftype != LLAMA_V2_FTYPE_ALL_F32     &&
            hparams.ftype != LLAMA_V2_FTYPE_MOSTLY_F16  &&
            hparams.ftype != LLAMA_V2_FTYPE_MOSTLY_Q8_0) {
            printf("\nLegacy LLAMA GGJT v1 compatability changes triggered.\n");
        }
    }

    if (file_version < LLAMA_V2_FILE_VERSION_GGJT_V3) {
        if (hparams.ftype == LLAMA_V2_FTYPE_MOSTLY_Q4_0 ||
            hparams.ftype == LLAMA_V2_FTYPE_MOSTLY_Q4_1 ||
            hparams.ftype == LLAMA_V2_FTYPE_MOSTLY_Q8_0) {
            printf("\nLegacy LLAMA GGJT v2 compatability changes triggered.\n");
        }
    }

    if (vocab_only) {
        return;
    }

    auto & ctx = model.ctx;

    size_t ctx_size;
    size_t mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    fprintf(stderr, "%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/1024.0/1024.0);

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_V2_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        const size_t mem_required =
            ctx_size +
            mmapped_size +
            MEM_REQ_SCRATCH0_2().at(model.type) +
            MEM_REQ_SCRATCH1_2().at(model.type) +
            MEM_REQ_EVAL_2().at(model.type);

        // this is the memory required by one llama_v2_state
        const size_t mem_required_state =
            scale*MEM_REQ_KV_SELF_2().at(model.type);

        fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);
    }

    // create the ggml context
    {
        lctx.model.buf.resize(ctx_size);
        if (use_mlock) {
            lctx.model.mlock_buf.init(lctx.model.buf.addr);
            lctx.model.mlock_buf.grow_to(lctx.model.buf.size);
        }

        struct ggml_v2_init_params params = {
            /*.mem_size   =*/ lctx.model.buf.size,
            /*.mem_buffer =*/ lctx.model.buf.addr,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_v2_init(params);
        if (!model.ctx) {
            throw format_old("ggml_v2_init() failed");
        }
    }

    // prepare memory for the weights
    {
        const uint32_t n_embd  = hparams.n_embd;
        const uint32_t n_layer = hparams.n_layer;
        const uint32_t n_vocab = hparams.n_vocab;

        ml->ggml_v2_ctx = ctx;

        model.tok_embeddings = ml->get_tensor("tok_embeddings.weight", {n_embd, n_vocab});
        model.norm           = ml->get_tensor("norm.weight",           {n_embd});
        model.output         = ml->get_tensor("output.weight",         {n_embd, n_vocab});

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            std::string layers_i = "layers." + std::to_string(i);

            layer.attention_norm = ml->get_tensor(layers_i + ".attention_norm.weight", {n_embd});

            layer.wq = ml->get_tensor(layers_i + ".attention.wq.weight", {n_embd, n_embd});
            layer.wk = ml->get_tensor(layers_i + ".attention.wk.weight", {n_embd, n_embd});
            layer.wv = ml->get_tensor(layers_i + ".attention.wv.weight", {n_embd, n_embd});
            layer.wo = ml->get_tensor(layers_i + ".attention.wo.weight", {n_embd, n_embd});

            layer.ffn_norm = ml->get_tensor(layers_i + ".ffn_norm.weight", {n_embd});

            layer.w1 = ml->get_tensor(layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff});
            layer.w2 = ml->get_tensor(layers_i + ".feed_forward.w2.weight", {  n_ff,   n_embd});
            layer.w3 = ml->get_tensor(layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff});
        }
    }

    ml->done_getting_tensors();

    // populate `tensors_by_name`
    for (llama_v2_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_v2_tensor);
    }

    ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &lctx.model.mlock_mmap : NULL);

    model.mapping = std::move(ml->mapping);
#if defined(GGML_USE_CUDA)
    {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));
        if(GetQuantsUnshuffled())
        {

        fprintf(stderr, "%s: [old cublas] offloading %d layers to GPU\n", __func__, n_gpu);

        size_t vram_total = 0;

        for (int i = 0; i < n_gpu; ++i) {
            const auto & layer = model.layers[i];

            ggml_v2_cuda_transform_tensor(layer.wq); vram_total += ggml_v2_nbytes(layer.wq);
            ggml_v2_cuda_transform_tensor(layer.wk); vram_total += ggml_v2_nbytes(layer.wk);
            ggml_v2_cuda_transform_tensor(layer.wv); vram_total += ggml_v2_nbytes(layer.wv);
            ggml_v2_cuda_transform_tensor(layer.wo); vram_total += ggml_v2_nbytes(layer.wo);
            ggml_v2_cuda_transform_tensor(layer.w1); vram_total += ggml_v2_nbytes(layer.w1);
            ggml_v2_cuda_transform_tensor(layer.w2); vram_total += ggml_v2_nbytes(layer.w2);
            ggml_v2_cuda_transform_tensor(layer.w3); vram_total += ggml_v2_nbytes(layer.w3);
        }
        if (n_gpu_layers > (int) hparams.n_layer) {
            fprintf(stderr, "%s: [old cublas] offloading output layer to GPU\n", __func__);
            ggml_v2_cuda_transform_tensor(model.output); vram_total += ggml_v2_nbytes(model.output);
        }

        fprintf(stderr, "%s: [old cublas] total VRAM used: %zu MB\n", __func__, vram_total / 1024 / 1024);
        }
        else
        {
            if(n_gpu>0)
            {
                printf("\n[WARNING: Old format does not support GPU offloading! It will be deactivated!]\n");
            }
        }
    }
#elif defined(GGML_USE_CLBLAST)
    {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));
        if(GetQuantsUnshuffled())
        {

        fprintf(stderr, "%s: [opencl] offloading %d layers to GPU\n", __func__, n_gpu);

        size_t vram_total = 0;

        for (int i = 0; i < n_gpu; ++i) {
            const auto & layer = model.layers[i];

            ggml_v2_cl_transform_tensor(layer.wq); vram_total += ggml_v2_nbytes(layer.wq);
            ggml_v2_cl_transform_tensor(layer.wk); vram_total += ggml_v2_nbytes(layer.wk);
            ggml_v2_cl_transform_tensor(layer.wv); vram_total += ggml_v2_nbytes(layer.wv);
            ggml_v2_cl_transform_tensor(layer.wo); vram_total += ggml_v2_nbytes(layer.wo);
            ggml_v2_cl_transform_tensor(layer.w1); vram_total += ggml_v2_nbytes(layer.w1);
            ggml_v2_cl_transform_tensor(layer.w2); vram_total += ggml_v2_nbytes(layer.w2);
            ggml_v2_cl_transform_tensor(layer.w3); vram_total += ggml_v2_nbytes(layer.w3);
        }
        if (n_gpu_layers > (int) hparams.n_layer) {
            fprintf(stderr, "%s: [opencl] offloading output layer to GPU\n", __func__);
            ggml_v2_cl_transform_tensor(model.output); vram_total += ggml_v2_nbytes(model.output);
        }

        fprintf(stderr, "%s: [opencl] total VRAM used: %zu MB\n", __func__, vram_total / 1024 / 1024);
        }
        else
        {
            if(n_gpu>0)
            {
                printf("\n[WARNING: Old format does not support GPU offloading! It will be deactivated!]\n");
            }
        }
    }
#else
    (void) n_gpu_layers;
#endif

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    lctx.t_load_us = ggml_v2_time_us() - lctx.t_start_us;
}

static bool llama_v2_model_load(
        const std::string & fname,
        llama_v2_context & lctx,
        int n_ctx,
        int n_gpu_layers,
        ggml_v2_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_v2_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        llama_v2_model_load_internal(fname, lctx, n_ctx, n_gpu_layers, memory_type, use_mmap, use_mlock,
                                  vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::string & err) {
        fprintf(stderr, "error loading model: %s\n", err.c_str());
        return false;
    }
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_v2_eval_internal(
        llama_v2_context & lctx,
    const llama_v2_token * tokens,
            const int   n_tokens,
            const int   n_past,
            const int   n_threads) {

    // enforce that the first token is BOS (not needed, messes with my context manip code)
    //if (n_past == 0 && tokens[0] != llama_v2_token_bos()) {
        //fprintf(stderr, "%s: first token must be BOS\n", __func__);
        // return false; //never fail. Not even in the face of Armageddon.
    //}

    const int64_t t_start_us = ggml_v2_time_us();

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = model.kv_self;

    LLAMA_V2_ASSERT(!!kv_self.ctx);

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_embd/hparams.n_head;

    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;

    struct ggml_v2_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.addr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_v2_context * ctx0 = ggml_v2_init(params);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    ggml_v2_cgraph gf = {};
    gf.n_threads = N >= 32 && ggml_v2_cpu_has_blas() && !ggml_v2_cpu_has_gpublas() ? 1 : n_threads;

    struct ggml_v2_tensor * embd = ggml_v2_new_tensor_1d(ctx0, GGML_V2_TYPE_I32, N);
    ggml_v2_set_name(embd, "embd");
    memcpy(embd->data, tokens, N*ggml_v2_element_size(embd));

    struct ggml_v2_tensor * inpL = ggml_v2_get_rows(ctx0, model.tok_embeddings, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_v2_tensor * inpSA = inpL;

        struct ggml_v2_tensor * cur;

        lctx.use_buf(ctx0, 0);

        // norm
        {
            cur = ggml_v2_rms_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_v2_mul(ctx0,
                        ggml_v2_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_v2_tensor * Qcur = ggml_v2_rope_inplace(ctx0, ggml_v2_reshape_3d(ctx0, ggml_v2_mul_mat(ctx0, model.layers[il].wq, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);
            struct ggml_v2_tensor * Kcur = ggml_v2_rope_inplace(ctx0, ggml_v2_reshape_3d(ctx0, ggml_v2_mul_mat(ctx0, model.layers[il].wk, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);
            ggml_v2_set_name(Qcur, "Qcur");
            ggml_v2_set_name(Kcur, "Kcur");

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                struct ggml_v2_tensor * Vcur = ggml_v2_transpose(ctx0, ggml_v2_reshape_2d(ctx0, ggml_v2_mul_mat(ctx0, model.layers[il].wv, cur), n_embd, N));

                struct ggml_v2_tensor * k = ggml_v2_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_v2_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_v2_tensor * v = ggml_v2_view_2d(ctx0, kv_self.v, N, n_embd,
                        (   n_ctx)*ggml_v2_element_size(kv_self.v),
                        (il*n_ctx)*ggml_v2_element_size(kv_self.v)*n_embd + n_past*ggml_v2_element_size(kv_self.v));

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(ctx0, Kcur, k));
                ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(ctx0, Vcur, v));
            }

            struct ggml_v2_tensor * Q =
                ggml_v2_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            ggml_v2_set_name(Q, "Q");

            struct ggml_v2_tensor * K =
                ggml_v2_permute(ctx0,
                        ggml_v2_reshape_3d(ctx0,
                            ggml_v2_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd, il*n_ctx*ggml_v2_element_size(kv_self.k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);
            ggml_v2_set_name(K, "K");

            // K * Q
            struct ggml_v2_tensor * KQ = ggml_v2_mul_mat(ctx0, K, Q);
            ggml_v2_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_v2_tensor * KQ_scale = ggml_v2_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head));
            ggml_v2_set_name(KQ_scale, "1/sqrt(n_embd/n_head)");

            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_v2_tensor * KQ_scaled = ggml_v2_scale_inplace(ctx0, KQ, KQ_scale);
            ggml_v2_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_v2_tensor * KQ_masked = ggml_v2_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            ggml_v2_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_v2_tensor * KQ_soft_max = ggml_v2_soft_max_inplace(ctx0, KQ_masked);
            ggml_v2_set_name(KQ_soft_max, "KQ_soft_max");


            // split cached V into n_head heads
            struct ggml_v2_tensor * V =
                ggml_v2_view_3d(ctx0, kv_self.v,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_v2_element_size(kv_self.v),
                        n_ctx*ggml_v2_element_size(kv_self.v)*n_embd/n_head,
                        il*n_ctx*ggml_v2_element_size(kv_self.v)*n_embd);
            ggml_v2_set_name(V, "V");

#if 1
            struct ggml_v2_tensor * KQV = ggml_v2_mul_mat(ctx0, V, KQ_soft_max);
            ggml_v2_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_v2_tensor * V_cont = ggml_v2_cpy(ctx0, V, ggml_v2_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd/n_head, n_head));
            struct ggml_v2_tensor * KQV = ggml_v2_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_v2_tensor * KQV_merged = ggml_v2_permute(ctx0, KQV, 0, 2, 1, 3);
            ggml_v2_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_v2_cpy(ctx0,
                    KQV_merged,
                    ggml_v2_new_tensor_2d(ctx0, GGML_V2_TYPE_F32, n_embd, N));
            ggml_v2_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_v2_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
        }

        lctx.use_buf(ctx0, 1);

        struct ggml_v2_tensor * inpFF = ggml_v2_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_v2_rms_norm(ctx0, inpFF);

                // cur = ffn_norm*cur
                cur = ggml_v2_mul(ctx0,
                        ggml_v2_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
            }

            struct ggml_v2_tensor * tmp = ggml_v2_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);

            cur = ggml_v2_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);

            // SILU activation
            cur = ggml_v2_silu(ctx0, cur);

            cur = ggml_v2_mul(ctx0, cur, tmp);

            cur = ggml_v2_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
        }

        cur = ggml_v2_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    lctx.use_buf(ctx0, 0);

    // used at the end to optionally extract the embeddings
    struct ggml_v2_tensor * embeddings = NULL;

    // norm
    {

        inpL = ggml_v2_rms_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_v2_mul(ctx0,
                    ggml_v2_repeat(ctx0, model.norm, inpL),
                    inpL);

        embeddings = inpL;
    }

    // lm_head
    inpL = ggml_v2_mul_mat(ctx0, model.output, inpL);

    lctx.use_buf(ctx0, -1);

    // logits -> probs
    //inpL = ggml_v2_soft_max_inplace(ctx0, inpL);

    // run the computation
    ggml_v2_build_forward_expand(&gf, inpL);
    ggml_v2_graph_compute       (ctx0, &gf);

#ifdef GGML_V2_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_V2_PERF to be defined
    ggml_v2_graph_print(&gf);
#endif

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_v2_graph_dump_dot(&gf, NULL, "llama.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_v2_get_data(inpL), sizeof(float)*n_vocab*N);

    // update kv token count
    lctx.model.kv_self.n = n_past + N;

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_v2_get_data(inpL), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_v2_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (!lctx.embedding.empty()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_v2_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_v2_used_mem(ctx0)/N;
    }

#if 0
    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
            ggml_v2_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif

    ggml_v2_free(ctx0);

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_v2_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (N > 1) {
        lctx.t_p_eval_us += ggml_v2_time_us() - t_start_us;
        lctx.n_p_eval += N;
    }

    return true;
}

//
// tokenizer
//

static size_t utf8_len2(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_v2_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llama_v2_sp_symbol>::value, "llama_v2_sp_symbol is not trivially copyable");

struct llama_v2_sp_bigram {
    struct comparator {
        bool operator()(llama_v2_sp_bigram & l, llama_v2_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_v2_sp_bigram>;
    using queue = std::priority_queue<llama_v2_sp_bigram, queue_storage, comparator>;
    llama_v2_sp_symbol::index left;
    llama_v2_sp_symbol::index right;
    float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_v2_tokenizer {
    llama_v2_tokenizer(const llama_v2_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_v2_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_v2_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len2(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    llama_v2_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_v2_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_v2_vocab & vocab_;
    std::vector<llama_v2_sp_symbol> symbols_;
    llama_v2_sp_bigram::queue work_queue_;
};

static std::vector<llama_v2_vocab::id> llama_v2_tokenize(const llama_v2_vocab & vocab, const std::string & text, bool bos) {
    llama_v2_tokenizer tokenizer(vocab);
    std::vector<llama_v2_vocab::id> output;

    if (text.empty()) {
        return output;
    }

    if (bos) {
        output.push_back(llama_v2_token_bos());
    }

    tokenizer.tokenize(text, output);
    return output;
}

//
// sampling
//

void llama_v2_sample_softmax(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates) {
    assert(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_v2_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_v2_token_data & a, const llama_v2_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}

void llama_v2_sample_top_k(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, int k, size_t min_keep) {
    const int64_t t_start_sample_us = ggml_v2_time_us();

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_v2_token_data & a, const llama_v2_token_data & b) {
            return a.logit > b.logit;
        };
        if (k == (int) candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}

void llama_v2_sample_top_p(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_v2_time_us();

    llama_v2_sample_softmax(ctx, candidates);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is greater than p or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}

void llama_v2_sample_tail_free(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    const int64_t t_start_sample_us = ggml_v2_time_us();

    llama_v2_sample_softmax(nullptr, candidates);

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);
    for (float & value : second_derivatives) {
        value /= second_derivatives_sum;
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}


void llama_v2_sample_typical(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_v2_time_us();

    // Compute the softmax of logits and calculate entropy
    llama_v2_sample_softmax(nullptr, candidates);

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_v2_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}

void llama_v2_sample_temperature(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_v2_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}

void llama_v2_sample_repetition_penalty(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, const llama_v2_token * last_tokens, size_t last_tokens_size, float penalty) {
    if (last_tokens_size == 0 || penalty == 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_v2_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        const auto * token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
        if (token_iter == last_tokens + last_tokens_size) {
            continue;
        }

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty;
        } else {
            candidates->data[i].logit /= penalty;
        }
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}

void llama_v2_sample_frequency_and_presence_penalties(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, const llama_v2_token * last_tokens_p, size_t last_tokens_size, float alpha_frequency, float alpha_presence) {
    if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_v2_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_v2_token, int> token_count;
    for (size_t i = 0; i < last_tokens_size; ++i) {
        token_count[last_tokens_p[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        int count = token_iter->second;
        candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
}


llama_v2_token llama_v2_sample_token_mirostat(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float tau, float eta, int m, float * mu) {
    assert(ctx);
    auto N = float(llama_v2_n_vocab(ctx));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_v2_time_us();

    llama_v2_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_v2_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
    llama_v2_token X = llama_v2_sample_token(ctx, candidates);
    t_start_sample_us = ggml_v2_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_v2_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return X;
}

llama_v2_token llama_v2_sample_token_mirostat_v2(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float tau, float eta, float * mu) {
    assert(ctx);
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_v2_time_us();

    llama_v2_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_v2_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    // Normalize the probabilities of the remaining words
    llama_v2_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
    llama_v2_token X = llama_v2_sample_token(ctx, candidates);
    t_start_sample_us = ggml_v2_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_v2_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    }
    return X;
}

llama_v2_token llama_v2_sample_token_greedy(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_v2_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_v2_token_data & a, const llama_v2_token_data & b) {
        return a.logit < b.logit;
    });

    llama_v2_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

llama_v2_token llama_v2_sample_token(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_v2_time_us();
    llama_v2_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    llama_v2_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_v2_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

//
// quantization
//

static void llama_v2_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, enum llama_v2_ftype ftype, int nthread) {
    ggml_v2_type quantized_type;
    switch (ftype) {
        case LLAMA_V2_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_V2_TYPE_Q4_0; break;
        case LLAMA_V2_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_V2_TYPE_Q4_1; break;
        case LLAMA_V2_FTYPE_MOSTLY_Q4_2: quantized_type = GGML_V2_TYPE_Q4_2; break;
        case LLAMA_V2_FTYPE_MOSTLY_Q4_3: quantized_type = GGML_V2_TYPE_Q4_3; break;
        case LLAMA_V2_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_V2_TYPE_Q5_0; break;
        case LLAMA_V2_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_V2_TYPE_Q5_1; break;
        case LLAMA_V2_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_V2_TYPE_Q8_0; break;
        default: throw format_old("invalid output file type %d\n", ftype);
    };

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    std::unique_ptr<llama_v2_model_loader> model_loader(new llama_v2_model_loader(fname_inp, /*use_mmap*/ false,
                                                                            /*vocab_only*/ false));
    llama_v2_file_saver file_saver(fname_out.c_str(), model_loader->file_loaders.at(0).get(), ftype);

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    std::mutex mutex;

    size_t idx = 0;
    for (llama_v2_load_tensor & tensor : model_loader->tensors_map.tensors) {
        llama_v2_buffer read_data;
        read_data.resize(tensor.size);
        tensor.data = read_data.addr;
        model_loader->load_data_for(tensor);

        printf("[%4zu/%4zu] %36s - %16s, type = %6s, ",
               ++idx, model_loader->tensors_map.tensors.size(),
               tensor.name.c_str(), llama_v2_format_tensor_shape(tensor.ne).c_str(),
               ggml_v2_type_name(tensor.type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = tensor.name.rfind("weight") == tensor.name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor.ne.size() == 2);

        // uncomment this to keep the output layer in FP16
        //if (tensor.name == "output.weight") {
        //    quantize = false;
        //}

        enum ggml_v2_type new_type;
        void * new_data;
        size_t new_size;
        llama_v2_buffer work;

        if (!quantize) {
            new_type = tensor.type;
            new_data = tensor.data;
            new_size = tensor.size;
            printf("size = %8.3f MB\n", tensor.size/1024.0/1024.0);
        } else {
            new_type = quantized_type;
            float * f32_data;
            size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
            llama_v2_buffer f32_conv_buf;
            if (tensor.type == GGML_V2_TYPE_F32) {
                f32_data = (float *) tensor.data;
            } else if (tensor.type == GGML_V2_TYPE_F16) {
                f32_conv_buf.resize(nelements * sizeof(float));
                f32_data = (float *) f32_conv_buf.addr;
                const auto * f16_data = (const ggml_v2_fp16_t *) tensor.data;
                for (size_t i = 0; i < nelements; i++) {
                    f32_data[i] = ggml_v2_fp16_to_fp32(f16_data[i]);
                }
            } else {
                throw format_old("type %s unsupported for integer quantization", ggml_v2_type_name(tensor.type));
            }

            printf("quantizing .. ");
            fflush(stdout);

            work.resize(nelements * 4); // upper bound on size
            new_data = work.addr;
            std::vector<int64_t> hist_cur(1 << 4, 0);

            int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2) {
                new_size = ggml_v2_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            } else {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements, chunk_size] () {
                    std::vector<int64_t> local_hist;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (!local_hist.empty()) {
                                for (int j=0; j<int(local_hist.size()); ++j) {
                                    hist_cur[j] += local_hist[j];
                                }
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        if (local_hist.empty()) {
                            local_hist.resize(hist_cur.size(), 0);
                        }
                        local_size += ggml_v2_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                if ((int) workers.size() < nthread_use - 1) {
                    workers.resize(nthread_use - 1);
                }
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it] = std::thread(compute);
                }
                compute();
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it].join();
                }
            }

            printf("size = %8.2f MB -> %8.2f MB | hist: ", tensor.size/1024.0/1024.0, new_size/1024.0/1024.0);
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
            }

            for (size_t i = 0; i < hist_cur.size(); i++) {
                printf("%5.3f ", hist_cur[i] / float(nelements));
            }
            printf("\n");
        }
        total_size_org += tensor.size;
        total_size_new += new_size;
        file_saver.write_tensor(tensor, new_type, new_data, new_size);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (size_t i = 0; i < hist_all.size(); i++) {
            printf("%5.3f ", hist_all[i] / float(sum_all));
        }
        printf("\n");
    }
}

//
// interface implementation
//

struct llama_v2_context * llama_v2_init_from_file(
                             const char * path_model,
            struct llama_v2_context_params   params) {
    ggml_v2_time_init();

    llama_v2_context * ctx = new llama_v2_context;

    if (params.seed < 0 || params.seed==0xFFFFFFFF) {
        params.seed = time(NULL);
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                ++*cur_percentage_p;
                fprintf(stderr, ".");
                fflush(stderr);
                if (percentage >= 100) {
                    fprintf(stderr, "\n");
                }
            }
        };
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_v2_type memory_type = params.f16_kv ? GGML_V2_TYPE_F16 : GGML_V2_TYPE_F32;

    if (!llama_v2_model_load(path_model, *ctx, params.n_ctx, params.n_gpu_layers, memory_type,
                          params.use_mmap, params.use_mlock, params.vocab_only,
                          params.progress_callback, params.progress_callback_user_data)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        llama_v2_free(ctx);
        return nullptr;
    }

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, ctx->model.hparams.n_ctx)) {
            fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
            llama_v2_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_v2_nbytes(ctx->model.kv_self.k) + ggml_v2_nbytes(ctx->model.kv_self.v);
            fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_vocab);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

        ctx->buf_compute.resize(MEM_REQ_EVAL_2().at(ctx->model.type));

        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0_2().at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1_2().at(ctx->model.type));
    }

    return ctx;
}

void llama_v2_free(struct llama_v2_context * ctx) {
    delete ctx;
}

int llama_v2_model_quantize(
        const char * fname_inp,
        const char * fname_out,
  enum llama_v2_ftype   ftype,
        int          nthread) {
    try {
        llama_v2_model_quantize_internal(fname_inp, fname_out, ftype, nthread);
        return 0;
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
        return 1;
    }
}

int llama_v2_apply_lora_from_file_internal(struct llama_v2_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    auto & model = ctx->model;

    const int64_t t_start_lora_us = ggml_v2_time_us();

    auto fin = std::ifstream(path_lora, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
        return 1;
    }

    // verify magic and version
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        uint32_t magic_ggla = 0x67676c61u; // 'ggla'
        if (magic != magic_ggla) {
            fprintf(stderr, "%s: bad file magic\n", __func__);
            return 1;
        }
        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != 1) {
            fprintf(stderr, "%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r;
    int32_t lora_alpha;
    fin.read((char *) &lora_r, sizeof(lora_r));
    fin.read((char *) &lora_alpha, sizeof(lora_alpha));
    float scaling = (float)lora_alpha / (float)lora_r;

    fprintf(stderr, "%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);


    // create a temporary ggml context to store the lora tensors
    // todo: calculate size from biggest possible tensor
    std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
    struct ggml_v2_init_params params;
    params.mem_size   = lora_buf.size();
    params.mem_buffer = lora_buf.data();
    params.no_alloc   = false;

    ggml_v2_context * lora_ctx = ggml_v2_init(params);
    std::unordered_map<std::string, struct ggml_v2_tensor *> lora_tensors;

    // create a name -> tensor map of the model to accelerate lookups
    std::unordered_map<std::string, struct ggml_v2_tensor*> model_tensors;
    for (auto & kv: model.tensors_by_name) {
        model_tensors.insert(kv);
    }


    // load base model
    std::unique_ptr<llama_v2_model_loader> model_loader;
    ggml_v2_context * base_ctx = NULL;
    llama_v2_buffer base_buf;
    if (path_base_model) {
        fprintf(stderr, "%s: loading base model from '%s'\n", __func__, path_base_model);
        model_loader.reset(new llama_v2_model_loader(path_base_model, /*use_mmap*/ true, /*vocab_only*/ false));

        size_t ctx_size;
        size_t mmapped_size;
        model_loader->calc_sizes(&ctx_size, &mmapped_size);
        base_buf.resize(ctx_size);

        ggml_v2_init_params base_params;
        base_params.mem_size   = base_buf.size;
        base_params.mem_buffer = base_buf.addr;
        base_params.no_alloc   = model_loader->use_mmap;

        base_ctx = ggml_v2_init(base_params);

        model_loader->ggml_v2_ctx = base_ctx;

        // maybe this should in llama_v2_model_loader
        if (model_loader->use_mmap) {
            model_loader->mapping.reset(new llama_v2_mmap(&model_loader->file_loaders.at(0)->file, /* prefetch */ false));
        }
    }

    // read tensors and apply
    bool warned = false;
    int n_tensors = 0;
    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
        if (fin.eof()) {
            break;
        }

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }

        std::string name;
        {
            char buf[1024];
            fin.read(buf, length);
            name = std::string(buf, length);
        }

        // check for lora suffix and get the type of tensor
        const std::string lora_suffix = ".lora";
        size_t pos = name.rfind(lora_suffix);
        if (pos == std::string::npos) {
            fprintf(stderr, "%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        std::string lora_type = name.substr(pos + lora_suffix.length());
        std::string base_name = name;
        base_name.erase(pos);
        // fprintf(stderr, "%s: %s => %s (lora type %s) ", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

        if (model_tensors.find(base_name) == model_tensors.end()) {
            fprintf(stderr, "%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
            return 1;
        }

        // create ggml tensor
        ggml_v2_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_V2_TYPE_F32;  break;
            case 1: wtype = GGML_V2_TYPE_F16;  break;
            default:
                    {
                        fprintf(stderr, "%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return false;
                    }
        }
        ggml_v2_tensor* lora_tensor;
        if (n_dims == 2) {
            lora_tensor = ggml_v2_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
        }
        else {
            fprintf(stderr, "%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }

        // load tensor data
        size_t offset = fin.tellg();
        size_t tensor_data_size = ggml_v2_nbytes(lora_tensor);
        offset = (offset + 31) & -32;
        fin.seekg(offset);
        fin.read((char*)lora_tensor->data, tensor_data_size);

        lora_tensors[name] = lora_tensor;

        // check if we have both A and B tensors and apply
        if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
            lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {

            ggml_v2_tensor * dest_t = model_tensors[base_name];
            ggml_v2_tensor * base_t;
            if (model_loader) {
                // load from base model
                if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
                    fprintf(stderr, "%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                    return 1;
                }
                size_t idx = model_loader->tensors_map.name_to_idx[base_name];
                llama_v2_load_tensor & lt = model_loader->tensors_map.tensors[idx];
                base_t = model_loader->get_tensor(base_name, { (uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1] });
                lt.data = (uint8_t *) lt.ggml_v2_tensor->data;
                model_loader->load_data_for(lt);
                lt.ggml_v2_tensor->data = lt.data;
            }
            else {
                base_t = dest_t;
            }

            if (ggml_v2_is_quantized(base_t->type)) {
                if (!warned) {
                    fprintf(stderr, "%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                                    "use a f16 or f32 base model with --lora-base\n", __func__);
                    warned = true;
                }
            }

            ggml_v2_tensor * loraA = lora_tensors[base_name + ".loraA"];
            ggml_v2_tensor * loraB = lora_tensors[base_name + ".loraB"];

            if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
                fprintf(stderr, "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                               " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
                return 1;
            }

            // w = w + BA*s
            ggml_v2_tensor * BA = ggml_v2_mul_mat(lora_ctx, loraA, loraB);

            if (scaling != 1.0f) {
                ggml_v2_tensor * scale_tensor = ggml_v2_new_f32(lora_ctx, scaling);
                BA = ggml_v2_scale_inplace(lora_ctx, BA, scale_tensor);
            }

            ggml_v2_tensor * r;
            if (base_t == dest_t) {
                r = ggml_v2_add_inplace(lora_ctx, dest_t, BA);
            }
            else {
                r = ggml_v2_add(lora_ctx, base_t, BA);
                r = ggml_v2_cpy(lora_ctx, r, dest_t);
            }

            struct ggml_v2_cgraph gf = ggml_v2_build_forward(r);
            gf.n_threads = n_threads;
            ggml_v2_graph_compute(lora_ctx, &gf);

            // we won't need these tensors again, reset the context to save memory
            ggml_v2_free(lora_ctx);
            lora_ctx = ggml_v2_init(params);
            lora_tensors.clear();

            n_tensors++;
            if (n_tensors % 4 == 0) {
                fprintf(stderr, ".");
            }
        }
    }

    // TODO: this should be in a destructor, it will leak on failure
    ggml_v2_free(lora_ctx);
    if (base_ctx) {
        ggml_v2_free(base_ctx);
    }

    const int64_t t_lora_us = ggml_v2_time_us() - t_start_lora_us;
    fprintf(stderr, " done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

int llama_v2_apply_lora_from_file(struct llama_v2_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return llama_v2_apply_lora_from_file_internal(ctx, path_lora, path_base_model, n_threads);
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__, err.c_str());
        return 1;
    }
}

int llama_v2_get_kv_cache_token_count(const struct llama_v2_context * ctx) {
    return ctx->model.kv_self.n;
}

#define LLAMA_V2_MAX_RNG_STATE (64*1024)

void llama_v2_set_rng_seed(struct llama_v2_context * ctx, int seed) {
    if (seed < 0 || seed==0xFFFFFFFF) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t llama_v2_get_state_size(const struct llama_v2_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = sizeof(size_t);
    const size_t s_rng             = LLAMA_V2_MAX_RNG_STATE;
    const size_t s_logits_capacity = sizeof(size_t);
    const size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embedding.size() * sizeof(float);
    const size_t s_kv_size         = sizeof(size_t);
    const size_t s_kv_ntok         = sizeof(int);
    const size_t s_kv              = ctx->model.kv_self.buf.size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_size
        + s_kv_ntok
        + s_kv
    );

    return s_total;
}

// Copies the state to the specified destination address
size_t llama_v2_copy_state_data(struct llama_v2_context * ctx, uint8_t * dst) {
    uint8_t * out = dst;

    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[LLAMA_V2_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, LLAMA_V2_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        memcpy(out, &rng_size,   sizeof(rng_size));    out += sizeof(rng_size);
        memcpy(out, &rng_buf[0], LLAMA_V2_MAX_RNG_STATE); out += LLAMA_V2_MAX_RNG_STATE;
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        memcpy(out, &logits_cap,  sizeof(logits_cap));  out += sizeof(logits_cap);
        memcpy(out, &logits_size, sizeof(logits_size)); out += sizeof(logits_size);

        if (logits_size) {
            memcpy(out, ctx->logits.data(), logits_size * sizeof(float));
        }

        out += logits_cap * sizeof(float);
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        memcpy(out, &embedding_size, sizeof(embedding_size)); out += sizeof(embedding_size);

        if (embedding_size) {
            memcpy(out, ctx->embedding.data(), embedding_size * sizeof(float));
            out += embedding_size * sizeof(float);
        }
    }

    // copy kv cache
    {
        const auto & kv_self = ctx->model.kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd;
        const int    n_ctx   = hparams.n_ctx;

        const size_t kv_size = kv_self.buf.size;
        const int    kv_ntok = llama_v2_get_kv_cache_token_count(ctx);

        memcpy(out, &kv_size, sizeof(kv_size)); out += sizeof(kv_size);
        memcpy(out, &kv_ntok, sizeof(kv_ntok)); out += sizeof(kv_ntok);

        if (kv_size) {
            const size_t elt_size = ggml_v2_element_size(kv_self.k);

            char buffer[4096];

            ggml_v2_context * cpy_ctx = ggml_v2_init({ sizeof(buffer), buffer, /* no_alloc */ true });
            ggml_v2_cgraph gf{};
            gf.n_threads = 1;

            ggml_v2_tensor * kout3d = ggml_v2_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            kout3d->data = out;
            out += ggml_v2_nbytes(kout3d);

            ggml_v2_tensor * vout3d = ggml_v2_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            vout3d->data = out;
            out += ggml_v2_nbytes(vout3d);

            ggml_v2_tensor * k3d = ggml_v2_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_v2_tensor * v3d = ggml_v2_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(cpy_ctx, k3d, kout3d));
            ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(cpy_ctx, v3d, vout3d));
            ggml_v2_graph_compute(cpy_ctx, &gf);

            ggml_v2_free(cpy_ctx);
        }
    }

    const size_t written  = out - dst;
    const size_t max_size = llama_v2_get_state_size(ctx);

    LLAMA_V2_ASSERT(written <= max_size);

    return written;
}

// Sets the state reading from the specified source address
size_t llama_v2_set_state_data(struct llama_v2_context * ctx, const uint8_t * src) {
    const uint8_t * inp = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[LLAMA_V2_MAX_RNG_STATE];

        memcpy(&rng_size,   inp, sizeof(rng_size));    inp += sizeof(rng_size);
        memcpy(&rng_buf[0], inp, LLAMA_V2_MAX_RNG_STATE); inp += LLAMA_V2_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        LLAMA_V2_ASSERT(rng_ss.fail() == false);
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  inp, sizeof(logits_cap));  inp += sizeof(logits_cap);
        memcpy(&logits_size, inp, sizeof(logits_size)); inp += sizeof(logits_size);

        LLAMA_V2_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
        }

        inp += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, inp, sizeof(embedding_size)); inp += sizeof(embedding_size);

        LLAMA_V2_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
            inp += embedding_size * sizeof(float);
        }
    }

    // set kv cache
    {
        const auto & kv_self = ctx->model.kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd;
        const int    n_ctx   = hparams.n_ctx;

        size_t kv_size;
        int kv_ntok;

        memcpy(&kv_size, inp, sizeof(kv_size)); inp += sizeof(kv_size);
        memcpy(&kv_ntok, inp, sizeof(kv_ntok)); inp += sizeof(kv_ntok);

        if (kv_size) {
            LLAMA_V2_ASSERT(kv_self.buf.size == kv_size);

            const size_t elt_size = ggml_v2_element_size(kv_self.k);

            char buffer[4096];

            ggml_v2_context * cpy_ctx = ggml_v2_init({ sizeof(buffer), buffer, /* no_alloc */ true });
            ggml_v2_cgraph gf{};
            gf.n_threads = 1;

            ggml_v2_tensor * kin3d = ggml_v2_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            kin3d->data = (void *) inp;
            inp += ggml_v2_nbytes(kin3d);

            ggml_v2_tensor * vin3d = ggml_v2_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            vin3d->data = (void *) inp;
            inp += ggml_v2_nbytes(vin3d);

            ggml_v2_tensor * k3d = ggml_v2_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_v2_tensor * v3d = ggml_v2_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(cpy_ctx, kin3d, k3d));
            ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(cpy_ctx, vin3d, v3d));
            ggml_v2_graph_compute(cpy_ctx, &gf);

            ggml_v2_free(cpy_ctx);
        }

        ctx->model.kv_self.n = kv_ntok;
    }

    const size_t nread    = inp - src;
    const size_t max_size = llama_v2_get_state_size(ctx);

    LLAMA_V2_ASSERT(nread <= max_size);

    return nread;
}

int llama_v2_eval(
        struct llama_v2_context * ctx,
           const llama_v2_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!llama_v2_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_v2_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    return 0;
}

int llama_v2_tokenize(
        struct llama_v2_context * ctx,
                  const char * text,
                 llama_v2_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = llama_v2_tokenize(ctx->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int llama_v2_n_vocab(const struct llama_v2_context * ctx) {
    return ctx->vocab.id_to_token.size();
}

int llama_v2_n_ctx(const struct llama_v2_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int llama_v2_n_embd(const struct llama_v2_context * ctx) {
    return ctx->model.hparams.n_embd;
}

float * llama_v2_get_logits(struct llama_v2_context * ctx) {
    return ctx->logits.data();
}

float * llama_v2_get_embeddings(struct llama_v2_context * ctx) {
    return ctx->embedding.data();
}

const char * llama_v2_token_to_str(const struct llama_v2_context * ctx, llama_v2_token token) {
    if (token >= llama_v2_n_vocab(ctx)) {
        return nullptr;
    }

    return ctx->vocab.id_to_token[token].tok.c_str();
}

llama_v2_token llama_v2_token_bos() {
    return 1;
}

llama_v2_token llama_v2_token_eos() {
    return 2;
}

llama_v2_token llama_v2_token_nl() {
    return 13;
}


void llama_v2_print_timings(struct llama_v2_context * ctx) {
    const int64_t t_end_us = ggml_v2_time_us();

    const int32_t n_sample = std::max(1, ctx->n_sample);
    const int32_t n_eval   = std::max(1, ctx->n_eval);
    const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
    fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per token)\n", __func__, 1e-3 * ctx->t_sample_us, n_sample, 1e-3 * ctx->t_sample_us / n_sample);
    fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
    fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token)\n", __func__, 1e-3 * ctx->t_eval_us,   n_eval,   1e-3 * ctx->t_eval_us   / n_eval);
    fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0);
}

void llama_v2_reset_timings(struct llama_v2_context * ctx) {
    ctx->t_start_us = ggml_v2_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * llama_v2_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_v2_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_v2_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_v2_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_v2_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_v2_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_v2_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_v2_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_v2_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_v2_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_v2_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_v2_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_v2_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_v2_cpu_has_sse3())        + " | ";
    s += "VSX = "         + std::to_string(ggml_v2_cpu_has_vsx())         + " | ";

    return s.c_str();
}

// For internal test use
std::vector<std::pair<std::string, struct ggml_v2_tensor *>>& llama_v2_internal_get_tensor_map(struct llama_v2_context * ctx) {
    return ctx->model.tensors_by_name;
}


// TODO: Calculate this constant from the vocabulary
#define MAX_TOKEN_LEN 18
// SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
std::vector<llama_v2_token> legacy_llama_v2_tokenize(const llama_v2_vocab & vocab, const std::string & text, bool bos) {
    std::vector<llama_v2_token> res;
    std::vector<int> score;
    std::vector<llama_v2_token> prev;
    int len = text.length();

    score.resize(len + 1);
    prev.resize(len + 1);

    // Forward pass
    for (int i = 0; i < len; i++) {
        int max_len = std::min(len - i, MAX_TOKEN_LEN);
        for (int sub_len = 1; sub_len <= max_len; sub_len++) {
            auto sub = text.substr(i, sub_len);
            auto token = vocab.token_to_id.find(sub);
            if (token != vocab.token_to_id.end()) {
                int token_score = sub.length() * sub.length();
                int local_score = score[i] + token_score;
                int next = i + sub_len;
                if (score[next] < local_score) {
                    score[next] = local_score;
                    prev[next] = (*token).second;
                }
            }
        }
    }

    // Backward pass
    int i = len;
    while (i > 0) {
        llama_v2_token token_id = prev[i];
        if (token_id == 0) {
	    // TODO: Return error or something more meaningful
            printf("failed to tokenize string!\n");
	    break;
        }
        res.push_back(token_id);
        auto token = vocab.id_to_token[token_id].tok;
        i -= token.length();
    }

    if (bos) {
        res.push_back(1); // TODO: replace with vocab.bos
    }

    // Pieces are in reverse order so correct that
    std::reverse(res.begin(), res.end());

    return res;
}

int legacy_llama_v2_tokenize(
        struct llama_v2_context * ctx,
                  const char * text,
                 llama_v2_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = legacy_llama_v2_tokenize(ctx->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

std::vector<llama_v2_token> legacy_llama_v2_tokenize(struct llama_v2_context * ctx, const std::string & text, bool add_bos) {
    std::vector<llama_v2_token> res(8096);
    int n = legacy_llama_v2_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    res.resize(n);

    return res;
}

std::vector<llama_token> llama_v2_tokenize(struct llama_v2_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int) add_bos);
    const int n = llama_v2_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}