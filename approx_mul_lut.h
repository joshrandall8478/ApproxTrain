
#ifndef APPROX_MUL_LUT_H_
#define APPROX_MUL_LUT_H_

#include <tensorflow/core/framework/op_kernel.h>
#include <fstream>
#include <string>
#include <vector>

typedef unsigned long long cudaTextureObject_t;

class approx_mul_lut_base {
public:
    explicit approx_mul_lut_base(tensorflow::OpKernelConstruction* context)
        : fp8_{false}, mant_width_{0}, mant_mask_{0}, a_shift_{0}, b_shift_{0}, lut_{false} {
        load_lut_binary(context);
    }

    virtual ~approx_mul_lut_base() = default;

    // Load LUT binary file
    void load_lut_binary(tensorflow::OpKernelConstruction* context) {
        std::string mant_lut_file_name;
        OP_REQUIRES_OK(context, context->GetAttr("mant_mul_lut", &mant_lut_file_name));
        OP_REQUIRES_OK(context, context->GetAttr("fp8", &fp8_));

        // OP_REQUIRES(context, !mant_lut_file_name.empty(),
        //             tensorflow::errors::InvalidArgument("No mant LUT file name given"));

        if (mant_lut_file_name.empty()) {
            lut_ = false;
            return;
        }
        if (!fp8_) {
            unsigned start_delimiter = mant_lut_file_name.find_last_of("_");
            unsigned stop_delimiter = mant_lut_file_name.find_last_of(".");
            OP_REQUIRES(context, start_delimiter != std::string::npos &&
                                    stop_delimiter != std::string::npos &&
                                    stop_delimiter > start_delimiter,
                        tensorflow::errors::InvalidArgument("Invalid LUT file name format"));
            auto mant_width_str = mant_lut_file_name.substr(start_delimiter + 1, stop_delimiter - start_delimiter - 1);
            mant_width_ = static_cast<uint8_t>(std::stoi(mant_width_str));
            a_shift_ = 23 - mant_width_ * 2;
            b_shift_ = 23 - mant_width_;
            mant_mask_ = ((1 << mant_width_) - 1) << (23 - mant_width_);
        } else {
            mant_width_ = 8; // Since FP8 values are 8 bits
            a_shift_ = 0;
            b_shift_ = 0;
            mant_mask_ = 0;
        }

        // Open mant mul file
        std::ifstream file(mant_lut_file_name, std::ios::in | std::ios::binary);
        OP_REQUIRES(context, file.is_open(),
                    tensorflow::errors::InvalidArgument("Failed to open LUT file: ", mant_lut_file_name));

        if (!fp8_) {
            size_t lut_size = static_cast<size_t>(1ULL << (mant_width_ * 2));
            mant_mul_lut_uint8_.resize(lut_size);
            file.read(
                reinterpret_cast<char*>(mant_mul_lut_uint8_.data()),
                mant_mul_lut_uint8_.size() * sizeof(uint8_t)
            );
            OP_REQUIRES(context, file.good(),
                        tensorflow::errors::InvalidArgument("Error reading LUT file: ", mant_lut_file_name));
        } else {
            size_t lut_size = 256 * 256 * 2; // Combined LUT with 512 entries
            mant_mul_lut_fp32_.resize(lut_size);
            file.read(
                reinterpret_cast<char*>(mant_mul_lut_fp32_.data()),
                mant_mul_lut_fp32_.size() * sizeof(float)
            );
            OP_REQUIRES(context, file.good(),
                        tensorflow::errors::InvalidArgument("Error reading LUT file: ", mant_lut_file_name));
        }
    }

    // Accessors
    cudaTextureObject_t& get_mant_mul_lut_text_() {
        return mant_mul_lut_text_;
    }

    void* get_mant_mul_lut_cuda() {
        if (fp8_) {
            return mant_mul_lut_cuda_fp32_;
        } else {
            return mant_mul_lut_cuda_uint8_;
        }
    }

    uint32_t get_mant_mask_() {
        return mant_mask_;
    }

    uint8_t get_a_shift_() {
        return a_shift_;
    }

    uint8_t get_b_shift_() {
        return b_shift_;
    }

    uint8_t get_mant_width_() {
        return mant_width_;
    }

    bool is_fp8() {
        return fp8_;
    }

protected:
    bool lut_;

    bool fp8_;
    std::vector<uint8_t> mant_mul_lut_uint8_;  // For 8-bit LUTs
    std::vector<float> mant_mul_lut_fp32_;     // Combined FP8 LUT

    uint8_t* mant_mul_lut_cuda_uint8_ = nullptr;
    float* mant_mul_lut_cuda_fp32_ = nullptr;
    cudaTextureObject_t mant_mul_lut_text_;

    std::string lut_file_name;
    uint8_t mant_width_;
    uint32_t mant_mask_;
    uint8_t a_shift_;
    uint8_t b_shift_;
};

template <typename Device>
class approx_mul_lut : public approx_mul_lut_base {
public:
    explicit approx_mul_lut(tensorflow::OpKernelConstruction* context);
    ~approx_mul_lut();

    cudaTextureObject_t& get_mant_mul_lut_text_() {
        return mant_mul_lut_text_;
    }
};

#endif // APPROX_MUL_LUT_H_
