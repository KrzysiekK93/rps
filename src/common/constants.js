export const IMAGE_WIDTH = 64
export const IMAGE_HEIGHT = 64
export const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
export const NUM_CLASSES = 3
export const NUM_DATASET_ELEMENTS = 2520
export const BYTES_PER_UINT8 = 4
export const BATCH_SIZE = 512

export const NUM_CHANNELS = 3

export const TRAIN_TEST_RATIO = 5 / 6
export const NUM_TRAIN_ELEMENTS = Math.floor(
  TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS
)
export const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS
