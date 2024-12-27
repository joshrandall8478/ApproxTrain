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
enum class AccumMode {
    RNE,
    RZ,
    FP16RNE,
    FP16RZ,
    BF16RNE,
    BF16RZ,
    SEARNE,
    SEAFP16RZ,
    SEABF16RZ,
    SEABF16
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
// string to float mode
inline FloatMode StringToFloatMode(const std::string& mode) {
    if (mode == "FP8E5M2") {
        return FloatMode::FP8E5M2;
    } else if (mode == "FP8HYB") {
        return FloatMode::FP8HYB;
    } else if (mode == "FP16") {
        return FloatMode::FP16;
    } else if (mode == "BF16") {
        return FloatMode::BF16;
    } else if (mode == "FP32") {
        return FloatMode::FP32;
    } else {
        return FloatMode::FP32;
    }
}
inline std::string AccumModeToString(AccumMode mode) {
    switch (mode) {
        case AccumMode::RNE:
            return "RNE";
        case AccumMode::RZ:
            return "RZ";
        case AccumMode::FP16RNE:
            return "FP16RNE";
        case AccumMode::FP16RZ:
            return "FP16RZ";
        case AccumMode::BF16RNE:
            return "BF16RNE";
        case AccumMode::BF16RZ:
            return "BF16RZ";
        case AccumMode::SEARNE:
            return "SEARNE";
        case AccumMode::SEAFP16RZ:
            return "SEAFP16RZ";
        case AccumMode::SEABF16RZ:
            return "SEABF16RZ";
        case AccumMode::SEABF16:
            return "SEABF16";
        default:
            return "Unknown AccumMode";
    }
}
// string to accum mode
inline AccumMode StringToAccumMode(const std::string& mode) {
    if (mode == "RNE") {
        return AccumMode::RNE;
    } else if (mode == "RZ") {
        return AccumMode::RZ;
    } else if (mode == "FP16RNE") {
        return AccumMode::FP16RNE;
    } else if (mode == "FP16RZ") {
        return AccumMode::FP16RZ;
    } else if (mode == "BF16RNE") {
        return AccumMode::BF16RNE;
    } else if (mode == "BF16RZ") {
        return AccumMode::BF16RZ;
    } else if (mode == "SEARNE") {
        return AccumMode::SEARNE;
    } else if (mode == "SEAFP16RZ") {
        return AccumMode::SEAFP16RZ;
    } else if (mode == "SEABF16RZ") {
        return AccumMode::SEABF16RZ;
    } else if (mode == "SEABF16") {
        return AccumMode::SEABF16;
    } else {
        return AccumMode::RNE;
    }
}
#endif // FLOATMODE_H