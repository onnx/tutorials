/**
 * SPDX-License-Identifier: Apache-2.0
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
