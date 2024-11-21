#ifndef FLOATMODE_H
#define FLOATMODE_H
#include <iostream>
#include <string>
enum class FloatMode {
    FP8E5M2,
    FP8HYB,
    FP16,
    BF16,
    FP32
};
inline std::string FloatModeToString(FloatMode mode) {
    switch (mode) {
        case FloatMode::FP8E5M2:
            return "FP8E5M2";
        case FloatMode::FP8HYB:
            return "FP8HYB";
        case FloatMode::FP16:
            return "FP16";
        case FloatMode::BF16:
            return "BF16";
        case FloatMode::FP32:
            return "FP32";
        default:
            return "Unknown FloatMode";
    }
}
#endif // FLOATMODE_H