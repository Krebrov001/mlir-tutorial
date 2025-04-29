#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
  MLIRContext ctx;

  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();

  // create OpBuilder
  OpBuilder builder(&ctx);
  auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());

  // Set the insertion point
  builder.setInsertionPointToEnd(mod.getBody());

  // create func
  auto i32 = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32, i32}, {i32});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test", funcType);

  // Add basic blocks
  auto entry = func.addEntryBlock();
  auto args = entry->getArguments();

  // Set the insertion point
  builder.setInsertionPointToEnd(entry);

  // create arith.addi
  auto addi = builder.create<arith::AddIOp>(builder.getUnknownLoc(), args[0], args[1]);

  // create func.return
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange({addi}));
  mod->print(llvm::outs());
  return 0;
}