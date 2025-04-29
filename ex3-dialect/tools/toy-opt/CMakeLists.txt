add_mlir_tool(ex3-opt toy-opt.cpp)
target_link_libraries(ex3-opt
  PRIVATE Toy
  MLIRIR MLIRParser MLIRSupport
  MLIROptLib        # correspond #include "mlir/Tools/mlir-opt/MlirOptMain.h"
  MLIRFuncDialect   # correspond #include "mlir/Dialect/Func/IR/FuncOps.h"
  MLIRTransforms    # correspond #include "mlir/Transforms/Passes.h"
)