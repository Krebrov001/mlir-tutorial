#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "toy/ToyDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"5
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

// mnemonic Refer to the name
class ToyOp<string mnemonic, list<Trait> traits = []> :
  Op<ToyDialect, mnemonic, traits>;

// Pure yes Trait，It means no SideEffect Pure functions of
def AddOp : ToyOp<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "add operation";
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($result)";
}

def SubOp : ToyOp<"sub", [Pure, SameOperandsAndResultType]> {
  let summary = "sub operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ConstantOp : ToyOp<"const", [Pure, InferTypeOpAdaptor]> {
  let summary = "const operation";
  let arguments = (ins APIntAttr:$value);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$value attr-dict";
  let extraClassDeclaration = [{
    int64_t getBitWidth() {
      return getResult().getType().getWidth();
    }
  }];
}

// Another implementation
//
// def ConstantOp : ToyOp<"const", [Pure, InferTypeOpInterface]> {
//   let summary = "const operation";
//   let arguments = (ins APIntAttr:$value);
//   let results = (outs AnyInteger:$result);
//   let assemblyFormat = "$value attr-dict";
//   let extraClassDeclaration = [{
//     static mlir::LogicalResult inferReturnTypes(
//       mlir::MLIRContext * context,
//       std::optional<::mlir::Location> location,
//       mlir::ValueRange operands,
//       mlir::DictionaryAttr attributes,
//       mlir::OpaqueProperties properties,
//       mlir::RegionRange regions,
//       llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes
//     );
//   }];
// }

def ReturnOp : ToyOp<"ret", [Terminator, ReturnLike]> {
  let summary = "return operation";
  let arguments = (ins AnyType:$data);
  let assemblyFormat = "$data attr-dict `:` type($data)";
}

def FuncOp : ToyOp<"func", [
  IsolatedFromAbove,
  FunctionOpInterface,
  /* Symbol, */ /* Symbol Will be automatically FunctionOpInterface Plus */
  /* CallableOpInterface, */ /* CallOpInterface Will be automatically FunctionOpInterface Plus */
  RegionKindInterface]> {
  let summary = "function";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    // FunctionOpInterface Need two Attr Come to record arg and res Name
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  dag regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = true;
  let extraClassDeclaration = [{
    // Method of FunctionOpInterface
    mlir::Region * getCallableRegion() {return &getBody();}
    // getFunctionType Functions will be generated automatically
    // mlir::FunctionType getFunctionType(); 

    // Method of CallableOpInterface
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {return getFunctionType().getInputs();}
    llvm::ArrayRef<mlir::Type> getResultTypes() {return getFunctionType().getResults();}

    // Method of RegionKindInterface
    static mlir::RegionKind getRegionKind(unsigned idx) { return mlir::RegionKind::SSACFG; }
  }];
}

def CallOp : ToyOp<"call", [CallOpInterface]> {
  let summary = "call operation";
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$arg_operands);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$callee `(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)";
  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() { return getCalleeAttr(); }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) { setCalleeAttr(callee.get<mlir::SymbolRefAttr>()); }
  }];
}

#endif
