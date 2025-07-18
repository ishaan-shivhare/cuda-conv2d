// conv2d_config.h
#pragma once

constexpr int BLOCK_SIZE = 16;
constexpr int K = 3;
constexpr int SH_TILE_W = BLOCK_SIZE + K - 1;
constexpr int IN_C = 3;
