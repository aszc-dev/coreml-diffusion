# Changelog

## [0.1.6](https://github.com/aszc-dev/coreml-diffusion/compare/v0.1.5...v0.1.6) (2026-06-13)


### 🐛 Bug Fixes

* **deps:** drop the stale &lt;3.13 Python cap (allow &gt;=3.12) ([9b44a5a](https://github.com/aszc-dev/coreml-diffusion/commit/9b44a5a4a4118a99c924ccb8301d4e1a300c3b01))

## [0.1.5](https://github.com/aszc-dev/coreml-diffusion/compare/v0.1.4...v0.1.5) (2026-06-13)


### ✨ Features

* **convert:** auto-detect model version from the checkpoint ([2a24d4e](https://github.com/aszc-dev/coreml-diffusion/commit/2a24d4efd196100dbdd0bf9d5dd61c6cce31d2ac))

## [0.1.4](https://github.com/aszc-dev/coreml-diffusion/compare/v0.1.3...v0.1.4) (2026-06-13)


### 🐛 Bug Fixes

* **convert:** generic LCM conversion for arbitrary checkpoints ([bedeb49](https://github.com/aszc-dev/coreml-diffusion/commit/bedeb49a84a33b4530c6c7d4ed4343a914a400e2))
* **inference:** add output_dtype to CoreMLTextEncoder ([eb6d3b5](https://github.com/aszc-dev/coreml-diffusion/commit/eb6d3b5b08b6ff34a4eb8683de3d1223fb447186))

## [0.1.3](https://github.com/aszc-dev/coreml-diffusion/compare/v0.1.2...v0.1.3) (2026-06-04)


### ✨ Features

* **convert:** add VAE and CLIP text-encoder conversion ([dc1f85b](https://github.com/aszc-dev/coreml-diffusion/commit/dc1f85bafe50d36655ff7ece0c052a30fd77bb81))
* **inference:** end-to-end Core ML pipeline (VAE + text-encoder swap) ([ca08b16](https://github.com/aszc-dev/coreml-diffusion/commit/ca08b16729529afbdf610d0e8ec2d09b849080c6))


### 🐛 Bug Fixes

* **inference:** expose .device on the Core ML adapters ([30a673e](https://github.com/aszc-dev/coreml-diffusion/commit/30a673eebe3927722214d0ab6a44fbc344d18f3a))


### 📚 Documentation

* **readme:** link the log.aszc.dev energy benchmark writeup ([77927b5](https://github.com/aszc-dev/coreml-diffusion/commit/77927b5dd5f1311a3b3c317692f3a347e3976a54))

## [0.1.2](https://github.com/aszc-dev/coreml-diffusion/compare/v0.1.1...v0.1.2) (2026-05-27)


### 🐛 Bug Fixes

* **attention:** convertible fp32 ORIGINAL attention for the Core ML GPU path ([#2](https://github.com/aszc-dev/coreml-diffusion/issues/2)) ([28e56fc](https://github.com/aszc-dev/coreml-diffusion/commit/28e56fcf8c2242ebbe4c05abd05f7e796069d7d1))
