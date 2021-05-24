import * as tf from '@tensorflow/tfjs'
import { IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS } from './constants'

const classNames = ['Rock', 'Paper', 'Scissors']

export const doSinglePrediction = async (model, img, options = {}) => {
  const resized = tf.tidy(() => {
    img = tf.browser.fromPixels(img)
    if (NUM_CHANNELS === 1) {
      const gray_mid = img.mean(2)
      img = gray_mid.expandDims(2)
    }
    const alignCorners = true
    return tf.image.resizeBilinear(
      img,
      [IMAGE_WIDTH, IMAGE_HEIGHT],
      alignCorners
    )
  })

  const logits = tf.tidy(() => {
    const batched = resized.reshape([
      1,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      NUM_CHANNELS
    ])

    return model.predict(batched)
  })

  const values = await logits.data()

  const { feedbackCanvas } = options
  if (feedbackCanvas) {
    await tf.browser.toPixels(resized.div(tf.scalar(255)), feedbackCanvas)
  }
  resized.dispose()
  logits.dispose()
  return classNames.map((className, idx) => ({
    className,
    probability: values[idx]
  }))
}

export const TFWrapper = model => {
  const calculateMaxScores = (scores, numBoxes, numClasses) => {
    const maxes = []
    const classes = []
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE
      let index = -1
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j]
          index = j
        }
      }
      maxes[i] = max
      classes[i] = index
    }
    return [maxes, classes]
  }

  const buildDetectedObjects = (
    width,
    height,
    boxes,
    scores,
    indexes,
    classes
  ) => {
    const count = indexes.length
    const objects = []
    for (let i = 0; i < count; i++) {
      const bbox = []
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j]
      }
      const minY = bbox[0] * height
      const minX = bbox[1] * width
      const maxY = bbox[2] * height
      const maxX = bbox[3] * width
      bbox[0] = minX
      bbox[1] = minY
      bbox[2] = maxX - minX
      bbox[3] = maxY - minY
      objects.push({
        bbox: bbox,
        class: classes[indexes[i]],
        score: scores[indexes[i]]
      })
    }
    return objects
  }

  const detect = input => {
    const batched = tf.tidy(() => {
      const img = tf.browser.fromPixels(input)
      return img.expandDims(0)
    })

    const height = batched.shape[1]
    const width = batched.shape[2]

    return model.executeAsync(batched).then(result => {
      const scores = result[0].dataSync()
      const boxes = result[1].dataSync()

      batched.dispose()
      tf.dispose(result)

      const [maxScores, classes] = calculateMaxScores(
        scores,
        result[0].shape[1],
        result[0].shape[2]
      )

      const prevBackend = tf.getBackend()
      tf.setBackend('cpu')
      const indexTensor = tf.tidy(() => {
        const boxes2 = tf.tensor2d(boxes, [
          result[1].shape[1],
          result[1].shape[3]
        ])
        return tf.image.nonMaxSuppression(
          boxes2,
          maxScores,
          20,
          0.5,
          0.5
        )
      })
      const indexes = indexTensor.dataSync()
      indexTensor.dispose()
      tf.setBackend(prevBackend)

      return buildDetectedObjects(
        width,
        height,
        boxes,
        maxScores,
        indexes,
        classes
      )
    })
  }
  return {
    detect: detect
  }
}