/**
 * SPDX-License-Identifier: Apache-2.0
 */

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

  var window: UIWindow?

  func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
    let window = UIWindow()
    window.rootViewController = CameraViewController()
    window.makeKeyAndVisible()

    self.window = window

    return true
  }
}

