#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char ** argv) {
  MLIRContext ctx;
  // first，Registration required dialect
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
  // Read indialect
  auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
  // Outputdialect
  src->print(llvm::outs());
  // Simple output，exist debug Commonly used when
  src->dump();
  return 0;
}