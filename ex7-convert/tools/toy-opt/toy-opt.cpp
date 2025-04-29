#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// Import Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// Import MLIR Bring your own Pass
#include "mlir/Transforms/Passes.h"
// Import our new one Dialect
#include "toy/ToyDialect.h"
#include "toy/ToyPasses.h"
using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  // register Dialect
  registry.insert<toy::ToyDialect, func::FuncDialect, arith::ArithDialect>();
  // registry.insert<LLVM::LLVMDialect>();
  // Register two Pass
  registerCSEPass();
  registerCanonicalizerPass();
  toy::registerPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
}
