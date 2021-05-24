import * as tf from '@tensorflow/tfjs'
import {
  IMAGE_SIZE,
  NUM_CLASSES,
  NUM_DATASET_ELEMENTS,
  NUM_CHANNELS,
  BYTES_PER_UINT8,
  NUM_TRAIN_ELEMENTS,
  NUM_TEST_ELEMENTS
} from './constants'

const RPS_IMAGES_SPRITE_PATH = '/data.png'
const RPS_LABELS_PATH = '/labels_uint8'

export class RPSDataset {
  constructor() {
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0
  }

  async load() {
    const img = new Image()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const imgRequest = new Promise((resolve, _reject) => {
      img.crossOrigin = ''
      img.onload = () => {
        img.width = img.naturalWidth
        img.height = img.naturalHeight

        const datasetBytesBuffer = new ArrayBuffer(
          NUM_DATASET_ELEMENTS * IMAGE_SIZE * BYTES_PER_UINT8 * NUM_CHANNELS
        )

        const chunkSize = Math.floor(NUM_TEST_ELEMENTS * 0.15)
        canvas.width = img.width
        canvas.height = chunkSize

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * chunkSize * IMAGE_SIZE * BYTES_PER_UINT8 * NUM_CHANNELS, 
            IMAGE_SIZE * chunkSize * NUM_CHANNELS
          )
          ctx.drawImage(
            img,
            0,
            i * chunkSize,
            img.width,
            chunkSize,
            0,
            0,
            img.width,
            chunkSize
          )
          
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
          let x = 0
          for (let j = 0; j < imageData.data.length; j += 4) {
            for (let i = 0; i < NUM_CHANNELS; i++) {
              datasetBytesView[x++] = imageData.data[j + i] / 255
            }
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer)
        resolve()
      }
      img.src = RPS_IMAGES_SPRITE_PATH
    })

    const labelsRequest = fetch(RPS_LABELS_PATH)
    const [_imgResponse, labelsResponse] = await Promise.all([
      imgRequest,
      labelsRequest
    ])

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer())

    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS)

    this.trainImages = this.datasetImages.slice(
      0,
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS * NUM_CHANNELS
    )
    this.testImages = this.datasetImages.slice(
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS * NUM_CHANNELS
    )
    this.trainLabels = this.datasetLabels.slice(
      0,
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    )
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS)
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length
        return this.trainIndices[this.shuffledTrainIndex]
      }
    )
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length
      return this.testIndices[this.shuffledTestIndex]
    })
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(
      batchSize * IMAGE_SIZE * NUM_CHANNELS
    )
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)

    for (let i = 0; i < batchSize; i++) {
      const idx = index()

      const startPoint = idx * IMAGE_SIZE * NUM_CHANNELS
      const image = data[0].slice(
        startPoint,
        startPoint + IMAGE_SIZE * NUM_CHANNELS
      )
      batchImagesArray.set(image, i * IMAGE_SIZE * NUM_CHANNELS)

      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      )
      batchLabelsArray.set(label, i * NUM_CLASSES)
    }
    const xs = tf.tensor3d(batchImagesArray, [
      batchSize,
      IMAGE_SIZE,
      NUM_CHANNELS
    ])
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])
    return { xs, labels }
  }
}
