/**
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) Facebook, Inc. and Microsoft Corporation.
 */

import UIKit
import MobileCoreServices
import Vision
import CoreML
import AVKit

class CameraViewController: UIViewController {

  var imageView: UIImageView!

  var captureSession: AVCaptureSession?
  let videoOutputQueue = DispatchQueue(label: "com.facebook.onnx.videoOutputQueue",
                                       qos: .userInitiated)

  var model = Model.Candy
  var modelExecutor: ModelExecutor?

  ///--------------------------------------
  // MARK: - View
  ///--------------------------------------

  override func loadView() {
    imageView = UIImageView()
    imageView.addGestureRecognizer(UITapGestureRecognizer(target: self,
                                                          action: #selector(switchToNextModel)))
    imageView.isUserInteractionEnabled = true
    imageView.contentMode = .scaleAspectFill
    self.view = imageView
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    setupExecutor(for: model)
    prepareCaptureSession()
  }

  ///--------------------------------------
  // MARK: - Actions
  ///--------------------------------------

  @objc func switchToNextModel() {
    // Capture model into local stack variable to make everything synchronized.
    let model = self.model.nextModel
    self.model = model

    // Stop the session and start it after we switch the model
    // All in all, this makes sure we switch fast and are not blocked by running the model.
    captureSession?.stopRunning()
    videoOutputQueue.async {
      self.modelExecutor = nil
      self.setupExecutor(for: model)

      DispatchQueue.main.async {
        self.captureSession?.startRunning()
      }
    }
  }

  ///--------------------------------------
  // MARK: - Setup
  ///--------------------------------------

  fileprivate func setupExecutor(for model: Model) {
    // Make sure we destroy existing executor before creating a new one.
    modelExecutor = nil

    // Create new one and store it in a var
    modelExecutor = try? ModelExecutor(for: model,
                                       executionHandler: (DispatchQueue.main, didGetPredictionResult))
  }

  fileprivate func prepareCaptureSession() {
    guard self.captureSession == nil else { return }

    let captureSession = AVCaptureSession()
    captureSession.sessionPreset = .hd1280x720

    let backCamera = AVCaptureDevice.default(for: .video)!
    let input = try! AVCaptureDeviceInput(device: backCamera)

    captureSession.addInput(input)

    let videoOutput = AVCaptureVideoDataOutput()
    videoOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)
    captureSession.addOutput(videoOutput)

    if let videoOutputConnection = videoOutput.connection(with: .video) {
      videoOutputConnection.videoOrientation = .portrait
    }

    captureSession.startRunning()

    self.captureSession = captureSession;
  }

  ///--------------------------------------
  // MARK: - Prediction
  ///--------------------------------------

  fileprivate func predict(_ pixelBuffer: CVPixelBuffer) {
    guard let modelExecutor = modelExecutor else {
      DispatchQueue.main.async {
        self.didGetPredictionResult(pixelBuffer: pixelBuffer, error: nil)
      }
      return
    }
    modelExecutor.execute(with: pixelBuffer)
  }

  fileprivate func didGetPredictionResult(pixelBuffer: CVPixelBuffer?, error: Error?) {
    guard let pixelBuffer = pixelBuffer else {
      print("Failed to get prediction result with error \(String(describing:error))")
      return
    }

    imageView.image = UIImage(ciImage: CIImage(cvPixelBuffer: pixelBuffer))
  }
}

///--------------------------------------
// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
///--------------------------------------

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

    predict(pixelBuffer)
  }
}
