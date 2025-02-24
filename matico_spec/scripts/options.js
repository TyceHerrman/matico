const {wasmPack} = require("esbuild-plugin-wasm-pack")
const {wasmLoader}= require('esbuild-plugin-wasm')

module.exports ={

  entryPoints:["./index.js"],
  outdir:"dist",
  format: "esm",
  bundle: true,
  plugins:[
    wasmLoader({mode:"embedded"}),
    wasmPack({
      path: "./",
      target:"bundler"
    })
  ],
}
