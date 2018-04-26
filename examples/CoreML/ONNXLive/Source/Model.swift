/**
 * Copyright (c) Facebook, Inc. and Microsoft Corporation.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation
import CoreML

enum Model {
  case Candy
  case Mosaic
  case RainPrincess
  case Udnie

  var MLModel: MLModel {
    switch self {
    case .Candy:
      return candy().model
    case .Mosaic:
      return mosaic().model
    case .RainPrincess:
      return rain_princess().model
    case .Udnie:
      return udnie().model
    }
  }

  var nextModel: Model {
    switch self {
    case .Candy:
      return .Mosaic
    case .Mosaic:
      return .RainPrincess
    case .RainPrincess:
      return .Udnie
    case .Udnie:
      return .Candy
    }
  }
}
