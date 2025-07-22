// conv2d_config.h
constexpr int BLOCK_SIZE = 16;
constexpr int K = 3;
constexpr int SH_TILE_W = BLOCK_SIZE + K - 1;
constexpr int IN_C = 4;
constexpr int TILE_W_PAD = ((SH_TILE_W + 3) / 4) * 4 / sizeof(float); // round up
