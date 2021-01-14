/**
 * SPDX-License-Identifier: Apache-2.0
 */

import Foundation
import CoreML
import Vision

final class ModelExecutor {

  typealias ExecutionHandler = (DispatchQueue, (CVPixelBuffer?, Error?) -> ())

  fileprivate let queue = DispatchQueue(label: "com.facebook.onnx.modelExecutor",
                                        qos: .userInitiated)
  fileprivate let vnModel: VNCoreMLModel
  fileprivate let vnRequest: VNCoreMLRequest

  init(for model: Model,
       executionHandler: ExecutionHandler) throws {
    self.vnModel = try VNCoreMLModel(for: model.MLModel)
    self.vnRequest = VNCoreMLRequest(model: vnModel, completionHandler: executionHandler)
    self.vnRequest.imageCropAndScaleOption = .centerCrop
  }

  func execute(with pixelBuffer: CVPixelBuffer) {
    queue.sync {
      let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
      try? handler.perform([ self.vnRequest ])
    }
  }

  func executeAsync(with pixelBuffer: CVPixelBuffer) {
    queue.async {
      let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
      try? handler.perform([ self.vnRequest ])
    }
  }
}

fileprivate extension VNCoreMLRequest {
  convenience init(model: VNCoreMLModel, completionHandler: ModelExecutor.ExecutionHandler) {
    self.init(model: model) { (request, error) in
      if let error = error {
        completionHandler.0.async {
          completionHandler.1(nil, error)
        }
        return
      }

      guard
        let results = request.results as? [VNPixelBufferObservation],
        let result = results.first
      else {
        // TODO: Error handling here
        return
      }

      completionHandler.0.async {
        completionHandler.1(result.pixelBuffer, nil)
      }
    }
  }
}
