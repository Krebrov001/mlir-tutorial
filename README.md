# Hands-On Practical MLIR Tutorial

Kexing Zhou（Zhou Kexing）

Beijing University

zhoukexing@pku.edu.cn

<!-- vscode-markdown-toc -->
* 1. [MLIR Introduction](#mlir-Introduction)
  * 1.1. [MLIR Compile pipeline](#mlir-Compile pipeline)
  * 1.2. [Common Dialect](#Common-dialect)
  * 1.3. [insight：“Timely optimization”](#insight：“Timely optimization”)
  * 1.4. [MLIR Uses](#mlir-Uses)
  * 1.5. [MLIR Disadvantages](#mlir-Disadvantages)
* 2. [MLIR Basic usage](#mlir-Basic usage)
  * 2.1. [IR Basic structure](#ir-Basic structure)
  * 2.2. [MLIR Basic project template](#mlir-Basic project template)
    * 2.2.1. [Configuration clangd](#Configuration-clangd)
  * 2.3. [MLIR Read in、Output](#mlir-Read in、Output)
  * 2.4. [Generate with code MLIR](#Generate with code-mlir)
* 3. [MLIR Op Structure of](#mlir-op-Structure of)
  * 3.1. [Attribute and Operand](#attribute-and-operand)
  * 3.2. [Attribute, Value and Type](#attribute,-value-and-type)
* 4. [MLIR Type conversion](#mlir-Type conversion)
  * 4.1. [Op Type conversion](#op-Type conversion)
  * 4.2. [Type / Attribute Type conversion](#type-/-attribute-Type conversion)
* 5. [MLIR The graph structure](#mlir-The graph structure)
  * 5.1. [MLIR Data flow graph structure](#mlir-Data flow graph structure)
  * 5.2. [MLIR Traversal and modification of data flow graph](#mlir-Traversal and modification of data flow graph)
  * 5.3. [MLIR Control flow diagram traversal and modification](#mlir-Control flow diagram traversal and modification)
* 6. [Basic Dialect project](#Basic-dialect-project)
  * 6.1. [TableGen Project templates](#tablegen-Project templates)
  * 6.2. [Tablegen Language Server](#tablegen-language-server)
  * 6.3. [IR Default definition and implementation of](#ir-Default definition and implementation of)
    * 6.3.1. [TableGen document](#tablegen-document)
    * 6.3.2. [Header file](#Header file)
    * 6.3.3. [Library files](#Library files)
    * 6.3.4. [Program entry](#Program entry)
* 7. [TableGen Op Definition explanation](#tablegen-op-Definition explanation)
  * 7.1. [Attribute、Type、Constraint](#attribute、type、constraint)
    * 7.1.1. [built-in Attribute](#built-in-attribute)
    * 7.1.2. [Built-in Type](#Built-in-type)
    * 7.1.3. [Why Attribute and Type All Constraint](#Why-attribute-and-type-All-constraint)
  * 7.2. [Verifier：DiscoverIRmistake](#verifier：Discoverirmistake)
    * 7.2.1. [emitError](#emiterror)
    * 7.2.2. [LogicalResult](#logicalresult)
  * 7.3. [Variadic：Variable parameters](#variadic：Variable parameters)
    * 7.3.1. [Multiple variable parameters：AttrSizedOperandSegments](#Multiple variable parameters：attrsizedoperandsegments)
  * 7.4. [AssemblyFormat：More readable output](#assemblyformat：More readable output)
    * 7.4.1. [Common keywords](#Common keywords)
    * 7.4.2. [additional attr dictionary](#additional-attr-dictionary)
    * 7.4.3. [Output type](#Output-type)
    * 7.4.4. [Optional output：Optional、UnitAttr](#Optional output：optional、unitattr)
  * 7.5. [Builder：Customize create function](#builder：Customize-create-function)
    * 7.5.1. [defaultBuilder](#defaultbuilder)
    * 7.5.2. [Customizebuilder](#Customizebuilder)
  * 7.6. [Custom functions](#Custom functions)
    * 7.6.1. [header target](#header-target)
  * 7.7. [use Trait](#use-trait)
    * 7.7.1. [Memory side effects：SideEffectInterfaces](#Memory side effects：sideeffectinterfaces)
    * 7.7.2. [Type inference：InferTypeOpInterface](#Type inference：infertypeopinterface)
  * 7.8. [function：FunctionOpTrait](#function：functionoptrait)
    * 7.8.1. [definition Return](#definition-return)
    * 7.8.2. [definition Function](#definition-function)
    * 7.8.3. [definition Call](#definition-call)
* 8. [Add to Pass](#Add to-pass)
  * 8.1. [Pass Project templates](#pass-Project templates)
  * 8.2. [Pass Definition explanation](#pass-Definition explanation)
    * 8.2.1. [designation Pass Which one is Op Run on](#designation-pass-Which one is-op-Run on)
    * 8.2.2. [With parameters Pass](#With parameters-pass)
  * 8.3. [Simple DCE Pass accomplish](#Simple-dce-pass-accomplish)
    * 8.3.1. [definition](#definition)
    * 8.3.2. [accomplish](#accomplish)
* 9. [Pattern Rewrite](#pattern-rewrite)
  * 9.1. [Pattern Rewrite](#pattern-rewrite-1)
    * 9.1.1. [describe Pattern](#describe-pattern)
    * 9.1.2. [Call Pattern](#Call-pattern)
    * 9.1.3. [Depedent Dialect & Linking](#depedent-dialect-&-linking)
  * 9.2. [Dialect Convertion (Type Conversion)](#dialect-convertion-(type-conversion))
    * 9.2.1. [TypeConverter](#typeconverter)
    * 9.2.2. [Conversion Pattern：Do it automatically Operand Type conversion](#conversion-pattern：Do it automatically-operand-Type conversion)
    * 9.2.3. [Details of type conversion and Debug](#Details of type conversion and-debug)
    * 9.2.4. [Use your own materialization](#Use your own-materialization)
  * 9.3. [use MLIR Already in it Pattern Do multi-step conversion](#use-mlir-Already in it-pattern-Do multi-step conversion)
* 10. [Customize Type](#Customize-type)
* 11. [TIPS](#tips)
  * 11.1. [How to find header files、Find the function you want](#How to find header files、Find the function you want)
  * 11.2. [How to find the library you need to connect to](#How to find the library you need to connect to)
  * 11.3. [How to speed up compilation](#How to speed up compilation)
  * 11.4. [go MLIR Copy code in](#go-mlir-Copy code in)
* 12. [MLIR Criticism：C++ v.s. Rust](#mlir-Criticism：c++-v.s.-rust)
* 13. [Issue & Reply](#issue-&-reply)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  1. <a name='mlir-Introduction'></a>MLIR Introduction

###  1.1. <a name='mlir-Compile pipeline'></a>MLIR Compile pipeline

MLIR Design a set of reusable compilation pipelines，Including reusable IR、Pass and IO system。exist IR middle，Multiple Dialect Can exist in mixture。MLIR Have defined a set Dialect Translation Graph：

![](fig/MLIR%20Dialects.jpg)

###  1.2. <a name='Common-dialect'></a>Common Dialect

MLIR of Dialect It is relatively independent，Here are some common ones dialect：

1. **func**：Processing functionsdialect，Contained function definitions、Call、Return and other basic operations
2. **arith**：Handle various operations such as addition, subtraction, multiplication, division, and shift
    * **math**：More complex operations，like log, exp, tan wait
3. **affine**：Handle loop nesting，Realize circular expansion、Polyhedral transformation and other algorithms
4. **scf**：(structured control flow) Structured control flow，reserve for，if Wait for statements
    * **cf**：Unstructured control flow，Only conditional jump command
5. **llvm**：LLVM IR of binding，Can be translated directly to LLVM Do subsequent compilation

MLIRCompilation from a high level tensor arrive Low-level scf,cf，Each stage is multiple dialect A mixture of，every time lowering Often only one dialect conduct。

<!-- ### MLIR Lowering process


Start with the following code，gradually lower arrive LLVM IR

```mlir
func.func @foo(%a: tensor<16x64xf64>, %b: tensor<16x64xf64>) -> tensor<16x64xf64> {
  %c = arith.addf %a, %b : tensor<16x64xf64>
  func.return %c : tensor<16x64xf64>
}
```

first，Will tensor Convert the operation to loop，The command is

```bash
mlir-opt \
  -convert-elementwise-to-linalg \
  -func-bufferize \
  -linalg-bufferize \
  -convert-linalg-to-affine-loops
```

```mlir
// affine
func.func @foo(%arg0: memref<16x64xf64>, %arg1: memref<16x64xf64>) -> memref<16x64xf64> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x64xf64>
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 64 {
      %0 = affine.load %arg0[%i, %j] : memref<16x64xf64>
      %1 = affine.load %arg1[%i, %j] : memref<16x64xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %alloc[%i, %j] : memref<16x64xf64>
    }
  }
  return %alloc : memref<16x64xf64>
}
```

Then -->

###  1.3. <a name='insight：“Timely optimization”'></a>insight：“Timely optimization”

Here is a brief example，dialect How it is mixed。

For example，I am pytorch，Some neural networks were generated，I want to represent these operations：

* Tensor It's a piece of belt shape Pointer：use **tensor** dialect
* Simple elementwise Addition, subtraction, multiplication and division：use **arith** dialect
* Complex log、exp Wait for operation：use **math** dialect
* Matrix linear algebra operation：use **linalg** dialect
* There may be some control flow：use **scf** dialect
* The entire network is a function：use **func** dialect

Next，Gradually lower arrive LLVM：

* Call Lowering Pass，Bundle **tensor** lowering arrive **linalg**，<u>And the others dialect Will not change</u>。
* Continue to call pass，Until **linalg** Convert to **affine** -> **scf** -> **cf**，<u>Others represent operations dialect Stay unchanged</u>。
* continue Lowering，Bundle **memref** Convert to bare pointer、**arith** and **func** Convert to llvm Built-in operations。
* at last，All non **llvm** dialect All were converted to **llvm** dialect，Now can be exported as llvm ir Leave it to llvm Continue to compile。

**visible，MLIR Compilation has a feature**：different dialect Is independent。

* For example，When doing loop expansion and other optimizations，I don't need to care about addition and subtraction that can be merged；When doing arithmetic expression optimization，Don't care about which function is currently in。
<!-- * In our use mlir When writing a new project，You can often use existing ones **arith** wait dialect Used to represent operations。 -->

**MLIR Can be optimized from all levels IR**：For example：

* exist **affine** Level，Can be expanded according to the loop size，Vectorization
* exist **scf**   Level，Can find loop invariant
* exist **arith** Level，Arithmetic identity can be used to optimize the code

MLIR of insight In“**Timely optimization**”。It's obvious，linalg level，It's easy to find that the matrix has been transposed twice，But once lower arrive scf，All transpose operations become loops，Optimization is difficult。

###  1.4. <a name='mlir-Uses'></a>MLIR Uses

We use MLIR，It's mainly to reuse the code that others have written，Generally included：

* Reuse already dialect As **enter**，Don't write the front end by yourself。
    * like Polygeist Can C Translated into Affine Dialect，So we don't have to write C Parser
* Will already have dialect **Mixed**or**As output**。
    * like arith wait dialect，Can be integrated directly，Don't need to write it yourself。
    * To generate binary When，Can be generated directly LLVM Dialect，Reuse backend LLVM Compile pipeline
* Reuse existing ones Pass。
    * Common Pass like CSE，DCE Can be reused
    * Dialect dedicated Pass，If the cycle expands，Can also be reused

###  1.5. <a name='mlir-Disadvantages'></a>MLIR Disadvantages

MLIR There are also disadvantages：

* Too bulky，Compilation、Long link time（Hundreds may be connectedMFiles）
  * Can be used lld To speed up links，But it's still slow [SeeTIPS](#113-How to speed up compilation)
* Dialect Extremely inflexible definition，More complex definition Op Very troublesome

##  2. <a name='mlir-Basic usage'></a>MLIR Basic usage

###  2.1. <a name='ir-Basic structure'></a>IR Basic structure

MLIR yes Tree structure，Each node is Operation，Op Can be composed Block，Block composition Region，and Region Can be nested in Op internal。

* **Operation** Refers to a single operation，Can be nested within the operation **Region**
* **Block** Refers to the basic block，The basic block contains one or more **Operation**
* **Region** Refers to the area，Similar to a loop body or a function body，Contains several **Block**

MLIR Use of basic blocks **“Basic block parameters”** To replace“phifunction”，As shown in the following example：

* **Block parameters**：Each basic block has parameters，Can be used in the block
* **Terminator**：Each basic block is usually a jump or a return，Block parameters need to be attached when jumping

```mlir
module {
func.func @foo(%a: i32, %b: i32, %c: i32) -> i32 {
  %cmp = arith.cmpi "sge", %a, %b : i32
  cf.cond_br %cmp, ^add(%a: i32), ^add(%b: i32)
^add(%1: i32):
  %ret = llvm.add %1, %c : i32
  cf.br ^ret
^ret:
  func.return %ret : i32
}
}
```

**module**: By default，mlir The outermost layer is `builtin.module`，As IR The root of。

###  2.2. <a name='mlir-Basic project template'></a>MLIR Basic project template

Build the first one mlir Projects are often very difficult，Here is a project template I use most commonly：

```
mlir-tutorial
├── install       # Install Prefix，Bundle MLIR Install here after compilation
├── llvm-project  # MLIR project
└── mlir-toy      # My own MLIR project
```

first，according to MLIR [getting started](https://mlir.llvm.org/getting_started/) Method，Install MLIR。Notice，Set it up during installation PREFIX for install Table of contents，As shown below，and getting start A slight difference on：

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 186a4b3b657878ae2aea23caf684b6e103901162 # The version used in this tutorial
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=/mlir-tutorial/install \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
```

exist build After that，Install to prefix

```bash
ninja install
```

Now，mlir Put all binary files、The library files are installed install In the directory。It's now possible `export PATH=/mlir-tutorial/install/bin:$PATH`，Convenient call `bin` The inside `mlir-opt` Wait for the program。

```bash
install
├── bin
├── examples
├── include
├── lib
└── share
```

Next，exist mlir-toy Create a simple project

```bash
mlir-toy
├── CMakeLists.txt
└── main.cpp
```

in CMakeLists.txt The file writing method is relatively fixed：

```cmake
cmake_minimum_required(VERSION 3.13.4)

project(mlir-toy VERSION 0.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # generate compile_commands.json Easy to highlight code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_executable(mlir-toy main.cpp)
```

exist main.cpp Write one in it main function，Then first build one time。Notice，Must be written `CMAKE_INSTALL_PREFIX`，socmake Can be found automatically MLIR。

```bash
cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=/mlir-tutorial/install
ninja
```

####  2.2.1. <a name='Configuration-clangd'></a>Configuration clangd

use vscode Default lint Tool running mlir Will be very stuck，Recommended to use clangd。

* Install in the extension clangd Plugin
* cmake When adding `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`（The aboveCMakeLists.txt Already added）
* Sometimes you want it compile_commands.json Copy to the project root directory，Or in vscode Configure it in the settings
* Once you find that the highlight explodes，vscodeinside Ctrl + Shift + P，enter clangd: restart language server
* sometimes，mlir Compilation options and clangd conflict，exist mlir-toy Created in the directory .clangd document，Remove related options：
    ```yaml
    CompileFlags:
      Remove:
        - -fno-lifetime-dse
    ```

###  2.3. <a name='mlir-Read in、Output'></a>MLIR Read in、Output

For testing mlir：

```mlir
func.func @test(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  func.return %c : i32
}
```

The easiest read-in output：

```cpp
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
  // Read into the file
  auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
  // Outputdialect，Can also output to llvm::errs(), llvm::dbgs()
  src->print(llvm::outs());
  // Simple output，exist debug Commonly used when
  src->dump();
  return 0;
}
```

Need to connect to all dependencies：

```cmake
target_link_libraries(
  ex1-io
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)
```

Test method：

```bash
./ex1-io ../ex1-io/ex1.mlir
```

###  2.4. <a name='Generate with code-mlir'></a>Generate with code MLIR

```cpp
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
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
```

**How to find builder.create Parameters of**：builder.create Internal call `Op::build` Functional，you can Ctrl + Mouse click to find `func::FuncOp` Definition，Then find the inside build function，Look at the parameter table。

##  3. <a name='mlir-op-Structure of'></a>MLIR Op Structure of

MLIR One of Operation It can contain some of the following things：

* Operand：this Op Accepted operands
* Result：this Op Generated new Value
* Attribute：It can be understood as a compiler constant
* Region：this Op Internal Region

MLIR middle，Attribute It is highly flexible，Allow insertion that does not exist attr，Allow different dialect Insert each other attribute。

###  3.1. <a name='attribute-and-operand'></a>Attribute and Operand

Attribute and Operand There are some differences。Attribute Refers to the amount known to the compiler，and Operand Refers to the amount that can only be known during runtime。

As the followingOp，0 It's one Attribute Not one Operand

```mlir
%c0 = arith.constant 0 : i32
```

###  3.2. <a name='attribute,-value-and-type'></a>Attribute, Value and Type

Value Must include Type，Type Can also be used as Attribute Attached in Operation superior。

For example functions Op，Although %a, %b Appears in the parameter table，But they are actually part of the function type，It's considered Type Attribute。

```mlir
func.func @test(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  func.return %c : i32
}
```

* `mlir-opt --mlir-print-op-generic` Let's print the code here，Get the following code。Parameter name is hidden，only function_type As attribute Keep it。
    ```mlir
    "builtin.module"() ({
      "func.func"() <{function_type = (i32, i32) -> i32, sym_name = "test"}> ({
      ^bb0(%arg0: i32, %arg1: i32):
        %0 = "arith.addi"(%arg0, %arg1) : (i32, i32) -> i32
        "func.return"(%0) : (i32) -> ()
      }) : () -> ()
    }) : () -> ()
    ```

##  4. <a name='mlir-Type conversion'></a>MLIR Type conversion

###  4.1. <a name='op-Type conversion'></a>Op Type conversion

MLIR All Op There is a unified storage format，Call `Operation`。`Operation` Saved inside OpName and all operands, results, attributes And other things。

User-defined `arith.addi` etc. Op，Essentially it's all `Operation` Pointer。But with `Operation*` The difference is，`AddIOp` Defined `Operation` How to interpret the data stored in it。like AddOp，I am one `Operation` Pointer，A function is also defined `getLhs` Used to return the first value，As lhs。

<center><img src='fig/op.svg' width='80%'></center>

**DownCast**：How to get it `Operation*` In the case of，Convert it to `AddOp` Woolen cloth？llvm Some conversion functions are provided，These functions check Operation of OpName，And convert。

```cpp
using namespace llvm;
void myCast(Operation * op) {
  auto res = cast<AddOp>(op); // Direct conversion，Failure to report an error
  auto res = dyn_cast<AddOp>(op); // Try to convert，Failed to return null，opfornullTimes error
  auto res = dyn_cast_if_present<AddOp>(op); // similar dyn_cast，opfornullReturn whennull
}
```

**Equal relationship**：two Operation* equal，Refers to them point to the same Operation Example，Not this Operation of operand,result,attr equal。

**Hashing**：Not modified IR In the case of，Each `Operation` Have a unique address。then，Can be used directly `Operation*` Create a Harbin table as value，Used to count IR Data or analysis：

```cpp
#include "llvm/ADT/DenseMap.h"
llvm::DenseMap<Operation*, size_t> numberOfReference;
```

###  4.2. <a name='type-/-attribute-Type conversion'></a>Type / Attribute Type conversion

MLIR of Type and Attribute and Op similar。Type Yes to TypeStorage Pointer，Attribute Also AttributeStorage Pointer。

* TypeStorage Will be stored inside Type Parameters of，like Integer Will save width，Array Will save Shape。

| Special pointer    | Universal pointer    | value（exist Contextmiddle）|
|:------------|:------------|:--------------------|
| AddOp       | Operation*  | Operation           |
| IntegerType | Type        | TypeStorage         |
| IntegerAttr | Attribute   | AttrStorage         |

**Global single case**：and Op The difference is，MLIR Context Will be completed Type and Attribute The work of deduplication。**Typeequal，TheirTypeStorageMust be equal, too。**

**DownCast**：Type of DownCast and Op same。

**Hashing**：and Op similar，Type Can also be used as Key Come and build a series of harbors，But not so often。

##  5. <a name='mlir-The graph structure'></a>MLIR The graph structure

MLIR inside，Two levels of diagram：

* The first one is Region Nested tree，This figure shows **Control flow**
* The second one is Op/Value The composition of the picture，This figure shows **Data flow**

###  5.1. <a name='mlir-Data flow graph structure'></a>MLIR Data flow graph structure

MLIR The data flow diagram is from Operation and Value Constructed。MLIR On the official website，IR Structure The inside [Two pictures](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-def-use-chains) Will MLIR The graph structure is explained very clearly：

First of all **Operation Connection**：

* Value Or from Operation of Result Or from BlockArgument
* Each Operation of Operand All are here Value Pointer
* To modify Operand When，The actual modification should be OpOperand

<center><img src='fig/DefUseChains.svg' width='80%'></center>

Then，yes **Value of use-chain**：

* Each Value All of them User Connected together

<center><img src='fig/Use-list.svg' width='80%'></center>

visible，MLIR The graph is a bidirectional graph structure，Be especially careful when traversing, especially when modifying。

* Modify OpOpeand When，correspond value of use-chain Will be secretly MLIR Change it
* Calling `value->getDefiningOp()` When，BlockArgument Will return null

###  5.2. <a name='mlir-Traversal and modification of data flow graph'></a>MLIR Traversal and modification of data flow graph

MLIR The traversal of data flow graphs often follows a pattern：Operation Call function to find Value，Use again Value Call function to find Operation，Alternate。

in，**Operation try to find Value Method**have：

* **getOperands**、**getResults**：These two are very common，As the following code can be used Op try to find Op
    ```cpp
    for(auto operand: op->getOperands()) {
      if(auto def = operand.getDefiningOp()) {
        // do something
      }
      else {
        // block argument
      }
    }
    ```
* **getOpOperands**：This needs to be changed operands Very useful when，For example, the following code will value Make a replacement：
    ```cpp
    IRMapping mapping;
    // Will op1 of results Map to op2 of results
    mapping.map(op1->getResults(), op2->getResults());
    for(auto &opOperand: op3->getOpOperands()) {
      // Will op3 The parameters contain op1 results Replaced by op2 of
      // lookupOrDefault It means it can't be found mapping Just use the original one
      opOperand.set(mapping.lookupOrDefault(opOperand.get()));
    }
    ```

**Value try to find Op Method**have：

* **getDefiningOp**：Possible to return null
* **getUses**：return OpOperand Iterator
* **getUsers**：return Operation Iterator

**OpofgetUsesandgetUser**：operation Yay getUses and getUsers function，Equivalent to this op All result of Uses or Users Put together。

**ValueModifications**：Value support **replaceAllUseWith** Revise，A sort of*Looks*The equivalent code is：
```cpp
for(auto & uses: value.getUses()) {
  uses.set(new_value);
}
```
But attention is needed，The above code is**Very dangerous**of。Because uses.set When，Will be modified value of use chain，and value of use-chain Being traversed，Maybe it'll hang up as soon as it is modified。then，Best used mlir Provide good `replaceAllUseWith` Come to modify。

###  5.3. <a name='mlir-Control flow diagram traversal and modification'></a>MLIR Control flow diagram traversal and modification

Compared with data flow graph，Control flow graph traversal is easier，Some commonly used functions：

* **op.getParentOp**, **op.getParentOfType**：Get FatherOp
* **op.getBlock**：Note that it is to return to the fatherblock，Not a functionblock
* **op.getBody**：This is the return to the inside block / region

Methods to traverse sons：

* **op.walk**：Recursively traverse all descendantsop：

    ```cpp
    // Recursively traverse all sons
    func.walk([](Operation * child) {
      // do something
    });
    // Recursively traverse all `ReturnOp` Type of son
    func.walk([](ReturnOp ret) {
      // do something
    })
    ```
    
* **block**：It's just one iterator，Can directly traverse：

    ```cpp
    Block * block = xxx
    for(auto & item: *block) {
      // do something
    }
    ```

Other traversal methods such as `getOps<xxx>` You can try it yourself。

The main purpose of modifying the control flow diagram `OpBuilder` Finish。Highly recommend finding `OpBuilder` Code，Take a look at all the functions inside，Common：

* **builder.create**：createop
* **builder.insert**：insertremoveofop
* **op->remove()**：Removed from the current block，But not deleted，Can be inserted into other blocks
* **op->erase()**：Removed from the current block，And delete

**Delete order**：Delete a op When，this op Cannot exist user，Otherwise, an error will be reported。

##  6. <a name='Basic-dialect-project'></a>Basic Dialect project

This section will talk about how to use it tablegen Define your own dialect，use mlir Comes with universal program portal `MlirOptMain`，generate `toy-opt`。

This is just a simple project template，`toy-opt` Only recognizable Op of generic Format。But don't worry，Let's build the project's skeleton first，Continuously adding feature。

```mlir
%c = "toy.add"(%a, %b): (i32, i32) -> i32 // Can be read
%c = toy.add %a, %b : i32 // Unable to read
```

###  6.1. <a name='tablegen-Project templates'></a>TableGen Project templates

This is too complicated，Please refer to the attached example `ex3-dialect`：

File structure：

```bash
ex3-dialect
├── CMakeLists.txt           # Controlling the other parts CMakeList
├── include
│   └── toy
│       ├── CMakeLists.txt  # control Dialect Defined CMakeList
│       ├── ToyDialect.h    # Dialect Header file
│       ├── ToyDialect.td   # Dialect TableGen document
│       ├── ToyOps.h        # Op Header file
│       ├── ToyOps.td       # Op TableGen document
│       └── Toy.td          # Bundle ToyDialect.td and ToyOps.td include Come together，For tablegen
├── lib
│   ├── CMakeLists.txt
│   └── toy.cpp             # Dialect library
└── tools
    └── toy-opt
        ├── CMakeLists.txt
        └── toy-opt.cpp     # Executable Tool
```

###  6.2. <a name='tablegen-language-server'></a>Tablegen Language Server

vscode supply mlir Extended，Can write for us tablegen Document help。exist `/mlir-tutorial/install/bin` in，have `mlir-lsp-server`。exist vscode Find in the settings mlir-lsp-server Settings，Set an absolute path，besides database The path。

Notice，lsp-server It's easy to suddenly collapse，Use it when it's exploded Ctrl+Shift+P，"mlir: restart language server"。

###  6.3. <a name='ir-Default definition and implementation of'></a>IR Default definition and implementation of

####  6.3.1. <a name='tablegen-document'></a>TableGen document

1. `include/ToyDialect.td`：definition Dialect Name andcppNamespace

    ```tablegen
    include "mlir/IR/OpBase.td"
    def ToyDialect : Dialect {
      let name = "toy";
      let cppNamespace = "::toy";
      let summary = "Toy Dialect";
    }
    ```

2. `include/ToyOps.td`：definition Operation

    ```tablegen
    include "mlir/IR/OpBase.td"
    include "toy/ToyDialect.td"
    include "mlir/Interfaces/SideEffectInterfaces.td"

    // mnemonic Refer to the name
    class ToyOp<string mnemonic, list<Trait> traits = []> :
      Op<ToyDialect, mnemonic, traits>;

    // Pure yes Trait，It means no SideEffect Pure functions of
    def AddOp : ToyOp<"add", [Pure]> {
      let summary = "add operation";
      let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
      let results = (outs AnyInteger:$result);
    }
    ```

3. `include/Toy.td`：Put the others td include Come together，Used to hand over tablegen generate

    ```tablegen
    include "toy/ToyDialect.td"
    include "toy/ToyOps.td"
    ```

   Pay attention to adding include Table of contents。
    ```cmake
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    ```

4. `include/CMakeLists.txt`：Call tablegen Generate code，in，The first Toy yes Dialect Name，The second one toy Refers to `toy.td`

    ```cmake
    add_mlir_dialect(Toy toy)
    ```

####  6.3.2. <a name='Header file'></a>Header file

5. tablegen The generated file is placed in `build/include/toy` inside，Includes default definitions and implementations

    * `ToyDialect.{h,cpp}.inc`：live Dialect Definition and implementation of
    * `Toy.{h,cpp}.inc`：live Op Definition and implementation of

    tablegen Generated `build` Table of contents，Need extra addition include
    ```cmake
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
    ```

6. `include/ToyDialect.h`：Bundle Dialect The definition of loading in

    ```cpp
    #pragma once
    #include "mlir/IR/BuiltinDialect.h"
    #include "toy/ToyDialect.h.inc" // include Just come in
    ```

7. `include/ToyOps.h`：Bundle Op The definition of loading in

    ```cpp
    #pragma once
    #include "mlir/IR/BuiltinOps.h"
    #include "mlir/IR/Builders.h"
    // td in include of，I want it here, too include Corresponding h document
    #include "toy/ToyDialect.h"
    #include "mlir/Interfaces/SideEffectInterfaces.h"
    #define GET_OP_CLASSES
    #include "toy/Toy.h.inc"
    ```

####  6.3.3. <a name='Library files'></a>Library files

8. `lib/toy.cpp`：Make default Dialect and Op The default implementation is loaded in

    ```cpp
    #include "toy/ToyDialect.h"
    #include "toy/ToyOps.h"
    #include "toy/ToyDialect.cpp.inc"
    #define GET_OP_CLASSES
    #include "toy/Toy.cpp.inc"
    using namespace toy;
    void ToyDialect::initialize() {
      // The following code will be generated Op List of，Specially used for initialization
      addOperations<
    #define GET_OP_LIST
    #include "toy/Toy.cpp.inc"
      >();
    }
    ```

9. `lib/CMakeLists.txt`：The front tablegen Will generate a `MLIRxxxIncGen` of Target，library Need to rely on this Target，Only by creating header files，Recompile toy.cpp。generally Library Named `MLIRToy` or `Toy`。
    ```cmake
    add_mlir_library(Toy toy.cpp DEPENDS MLIRToyIncGen)
    ```

####  6.3.4. <a name='Program entry'></a>Program entry

10. `tools/toy-opt/toy-opt.cpp`：mlir Provides a reusable and universal program entry，We can `MlirOptMain` Register before what we want Dialect and Pass，Call next `MlirOptMain`，You can use some of the default functions。
    
    ```cpp
    #include "mlir/IR/DialectRegistry.h"
    #include "mlir/Tools/mlir-opt/MlirOptMain.h"
    // Import Func Dialect
    #include "mlir/Dialect/Func/IR/FuncOps.h"
    // Import MLIR Bring your own Pass
    #include "mlir/Transforms/Passes.h"
    // Import our new one Dialect
    #include "toy/ToyDialect.h"
    using namespace mlir;
    using namespace llvm;

    int main(int argc, char ** argv) {
      DialectRegistry registry;
      // register Dialect
      registry.insert<toy::ToyDialect, func::FuncDialect>();
      // Register two Pass
      registerCSEPass();
      registerCanonicalizerPass();
      return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
    }
    ```

    Notice，You need to connect to this `MLIROptLib` Library：exist `tools/toy-opt/CMakeLists.txt` inside

    ```cmake
    add_mlir_tool(toy-opt toy-opt.cpp)
    target_link_libraries(toy-opt
      PRIVATE
      MLIRIR MLIRParser MLIRSupport
      Toy               # correspond #include "toy/ToyDialect.h"
      MLIROptLib        # correspond #include "mlir/Tools/mlir-opt/MlirOptMain.h"
      MLIRFuncDialect   # correspond #include "mlir/Dialect/Func/IR/FuncOps.h"
      MLIRTransforms    # correspond #include "mlir/Transforms/Passes.h"
    )
    ```

11. Simple use `ninja toy-opt`
    * `./toy-opt --help` Can write a document，There should be something inside cse and canonicalize two pass
    * `./toy-opt ../ex3-dialect/ex3.mlir` Read the file
    * `./toy-opt -canonicalize ../ex3-dialect/ex3-cse.mlir`，Can do it dce
    * `./toy-opt -cse ../ex3-dialect/ex3-cse.mlir`，Can do it cse

Why mlir Know ours Op Can be CSE and DCE Woolen cloth，Because we give Op Tagged `Pure` Trait，This means this Op It's a pure function。`Pure` Trait It will automatically register for us Op of CSE and DCE model。

##  7. <a name='tablegen-op-Definition explanation'></a>TableGen Op Definition explanation

Previous section introduced MLIR The skeleton of the project，Now we'll add bricks to it，let IR Definition、enter、Simpler output。

###  7.1. <a name='attribute、type、constraint'></a>Attribute、Type、Constraint

Add to Attribute Methods and Operand similar，All written in arguments in。

```tablegen
def ConstantOp : ToyOp<"const", [Pure]> {
  let summary = "const operation";
  let arguments = (ins APIntAttr:$value);
  let results = (outs AnyInteger:$result);
}
```

####  7.1.1. <a name='built-in-attribute'></a>built-in Attribute

See `mlir/IR/CommonAttrConstraints.td`，Commonly used：

* `TypeAttrOf<Type>`：Put one Type As Attr
* `FlatSymbolRefAttr`：call When the function，Function name Attr
* `SymbolNameAttr`: When defining a function，Function name Attr
* `UnitAttr`：Indicates abool，fortrueWhen，It's in attr Inside the table，for false Not here when
* `I64SmallVectorArrayAttr`：Integer array Attr，The difference from other integer arrays is，It uses SmallVector，It will be easier to use

####  7.1.2. <a name='Built-in-type'></a>Built-in Type

See `mlir/IR/CommonTypeConstraint.td`，Commonly used：

* `I1`, `I8`, `I16`, `I32`, `I64`, `I128`
* `AnyType`：Indicates any type
* `AnyInteger`：Represents any integer

####  7.1.3. <a name='Why-attribute-and-type-All-constraint'></a>Why Attribute and Type All Constraint

for Op Define a Attribute When，Actually specified [Operation](#41-op-Type conversion) in operands, results, attributes etc. How to explain。

picture Attribute、Type Such Shown “1. i One location operand Only interpreted as integers”、“1. j One location attr Only interpreted asSymbol” Agreement，It's limited to each field How to explain，Being regarded as “Constraint”。

###  7.2. <a name='verifier：Discoverirmistake'></a>Verifier：DiscoverIRmistake

exist tablegen Add it inside `hasVerifier=true`

```mlir
def SubOp : ToyOp<"sub", [Pure]> {
  let summary = "sub operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  let hasVerifier = true;
}
```

Then in `toy.cpp` Write in verifier Implementation：

```cpp
using namespace mlir;
LogicalResult SubOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return this->emitError() << "Lhs Type " << getLhs().getType()
      << " not equal to rhs " << getRhs().getType(); 
  return success();
}
```

####  7.2.1. <a name='emiterror'></a>emitError

emitError yes Op Functions with。MLIRin Op All will bring `emitError` function，Used to distinguish which one isOpAn error occurred。Here，us verify It's yourself，Just call your own `emitError` function。

* besides `emitWarning`，Can output Warning。

####  7.2.2. <a name='logicalresult'></a>LogicalResult

MLIR use LogicalResult Used to represent similar bool Value of，Its characteristics are：

* mlirSome other types of LogicalResult，As above emitError It can be automatically converted
* use success(), failure() generate true and false
* use succeed(x), failed(x) To determine whether it is true, false

###  7.3. <a name='variadic：Variable parameters'></a>Variadic：Variable parameters

use `Variadic<Type>` To describe variable parameters：

```tablegen
def AddOp : ToyOp<"add", [Pure]> {
  let summary = "add operation";
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
}
```

use `Option<Type>` To describe optional parameters：

```tablegen
def ReturnOp : ToyOp<"return", [Terminator, ReturnLike]> {
  let summary = "return operation"
  let arguments = (ins Optional<AnyInteger>:$data);
}
```

####  7.3.1. <a name='Multiple variable parameters：attrsizedoperandsegments'></a>Multiple variable parameters：AttrSizedOperandSegments

When a function has only one `Variadic` or `Optional` When，You can infer how many variable parameters are there based on the total number of parameters。But if there are multiple `Variadic` or `Optional`，Need to add `AttrSizedOperandSegments` Trait，this trait Will be Op Add one attribute Used to record whether each variable parameter exists，If there are many。

There are also related to this `AttrSizedResultSegments` Used to return multiple variable parameters，They're all in `OpBase.td` in。

###  7.4. <a name='assemblyformat：More readable output'></a>AssemblyFormat：More readable output

example：

```tablegen
def AddOp : ToyOp<"add", [Pure]> {
  let summary = "add operation";
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}
```

This will generate the following more readable code：

```mlir
%0 = toy.add %a, %b : i32, i32 -> i32
```

####  7.4.1. <a name='Common keywords'></a>Common keywords

* `$xxx` Used to represent operand or attribute
* `type($xxx)` Used to represent xxx Types of。
* ``` `keyword` ```： insert keyword
* `functional-type($inputs, results)`，Generate form `(i32, i32) -> i32` Function type
* `attr-dict`：Indicates additional attr dictionary。

####  7.4.2. <a name='additional-attr-dictionary'></a>additional attr dictionary

mlir Allowed as OP Insert any attribute，Allow cross dialect insert attribute。so，In definition op When，Always have to `attr-dict` Plus，This way, others inserted it attr Can save it too。

####  7.4.3. <a name='Output-type'></a>Output type

All dead without restrictions（AnyXXX，like AnyInteger）of operand，All need to be written clearly type，Or use `type($xxx)`，Or use `functional-type`。

####  7.4.4. <a name='Optional output：optional、unitattr'></a>Optional output：Optional、UnitAttr

against Optional and UnitAttr，MLIR Provides a Conditional grouping Syntax of：As shown below HWReg

```tablegen
def HWRegOp : ToyOp<"reg"> {
  let summary = "hardware register";
  let arguments = (ins I1:$clock, AnyInteger:$input, Optional<I1>:$reset, UnitAttr:$is_public);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = [{
    (`public` $is_public^)? $input
    `clock` $clock
    (`reset` $reset^)?
    attr-dict `:` functional-type($input, results)
  }];
}
```

There may be the following effects：

```mlir
%a = toy.reg public %b clock %clk reset %reset : (i32) -> i32
%a = toy.reg %b clock %clk reset %reset : (i32) -> i32
%a = toy.reg %b clock %clk : (i32) -> i32
```

* `[{xxx}]`，MLIRLong text in it can be used `[{}]` Cover it up。
* ``(`reset` $reset^)?``，in `(...)?` Indicates grouping，`^` Express judgment basis。Only corresponding `Optional` or `UnitAttr` When it exists，This group will be output。

###  7.5. <a name='builder：Customize-create-function'></a>Builder：Customize create function

Builder Will be here `builder.create<XXXOp>()` Called when，A simpler builder Can create Op Faster。

####  7.5.1. <a name='defaultbuilder'></a>defaultBuilder

MLIR Some will be generated by defaultbuilder。default builder Will ask to pass in first result Types of，Pass in again operand，attribute Value of。

```cpp
build(
  $_builder, $_state,
  mlir::Type Res1Type, mlir::Type Res2Type, ...,
  mlir::Value arg1, mlir::Value arg2, ...
)
```

sometimes，Some commonly used Attr，like `StringAttr`，MLIR It will be automatically generated `StringRef` for parameters builder，Used to facilitate call。Users don't need to use it `builder.getStringAttr(xxx)` Put it first `StringRef` Convert to `StringAttr` Let's pass the parameters again。

[Before](#711-built-in-attribute) Referred `I64SmallVectorArrayAttr` You can just pass one `SmallVector<int64_t>`，No need to pass one Attr Go in，It will be very convenient。

####  7.5.2. <a name='Customizebuilder'></a>Customizebuilder

For example，We can create Op When，The type of result is inferred：

```mlir
def SubOp : ToyOp<"sub", [Pure]> {
  let summary = "sub operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  let builders = [
    OpBuilder<
      (ins "mlir::Value":$lhs, "mlir::Value":$rhs),
      "build($_builder, $_state, lhs.getType(), lhs, rhs);"
    >
  ];
  let hasVerifier = true;
}
```

* first，mlir It will be generated automatically for us `build($_builder, $_state, ResultType, LhsValue, RhsValue)` of builder
* Our builder pass `lhs.getType()` infer result Types of，And call mlir Generate good builder，Implement automatic inference type

If only to infer the type，Recommended to use MLIR Specially implemented for type inference Trait: InferTypeOpInterface [An introduction later](#772-Type inferenceinfertypeopinterface)。

###  7.6. <a name='Custom functions'></a>Custom functions

tablegen Allow users to Op Add custom functions，For example，I want to get it directly ConstantOp The bit width of the type：

```tablegen
def ConstantOp : ToyOp<...> {
  let extraClassDeclaration = [{
    int64_t getBitWidth() {
      return getResult().getType().getWidth();
    }
  }];
}
```

so，When you want to get the bit width later，It can be more concise：

```cpp
auto w = op.getResult().getType().getWidth();
auto w = op.getBitWidth();
```

because tablegen No inside cpp Syntax Completion，Can only be in tablegen Write a method definition，Then in `toy.cpp` Write implementation inside（use `ninja MLIRToyIncGen` Generate header file）

```tablegen
def ConstantOp : ToyOp<...> {
  let extraClassDeclaration = [{
    int64_t getBitWidth();
  }];
}
```

####  7.6.1. <a name='header-target'></a>header target

one trick Yes `CMakeLists.txt` Add one in it target，This is changed every time tablegen document，Just need `ninja header` You can generate header files。

```cmake
add_custom_target(header DEPENDS MLIRToyIncGen)
```

###  7.7. <a name='use-trait'></a>use Trait

[Front](#634-Program entry) Introduction，Giving Op Mark on `Pure` after，Will be automatically cse, dce Pass understand。Apart from Pure Trait Outside，MLIR Provide us with a lot of useful things Trait，Here are the commonly used ones SideEffect，InferType and More complex and related to functions Trait。

use Trait Pay attention when：

1. Interface Users may be required to implement some fixed interfaces，trait Some inside `InterfaceMethod` There is no default implementation。
2. exist td Want to include trait of td document，exist h Want it in the middle include Corresponding h document

####  7.7.1. <a name='Memory side effects：sideeffectinterfaces'></a>Memory side effects：SideEffectInterfaces

`mlir/Interfaces/SideEffectInterfaces.{td,h}` The memory side effects are defined in the file interface

* **Pure**：Pure functions，It can be automatically cse，dce
* `MemRead`, `MemWrite`, `MemAlloc`, `MemFree`：Memory function

####  7.7.2. <a name='Type inference：infertypeopinterface'></a>Type inference：InferTypeOpInterface

`mlir/Interfaces/InferTypeOpInterface.{td,h}` Type inference is defined in the file Interface，Use type inference，you can：

* exist `assemblyFormat` You can write a few less type，The format written is more beautiful
* Automatically generate type inference builder，You only need to pass parameters to infer the return value type

Type inference mainly includes the following commonly used Trait：

* **SameOperandsAndResultType**：The operand and return value have the same type，After use assemblyFormat You only need to write any operand type
* **InferTypeOpInterface**：By entering and attr Type infer return value type，Write your own inference function
* **InferTypeOpAdaptor**：Similar to the previous one，But encapsulated Adaptor，It will be easier to write

Recommended use **InferTypeOpAdaptor**：
* exist tablegen in
    ```tablegen
    def ConstantOp : ToyOp<"const", [Pure, InferTypeOpAdaptor]> {
      let summary = "const operation";
      let arguments = (ins APIntAttr:$value);
      let results = (outs AnyInteger:$result);
      let assemblyFormat = "$value attr-dict"; // No need to write here type($result) It's
    }
    ```
* exist `toy.cpp` in
    ```cpp
    mlir::LogicalResult ConstantOp::inferReturnTypes(
      mlir::MLIRContext * context,
      std::optional<mlir::Location> location,
      Adaptor adaptor,
      llvm::SmallVectorImpl<mlir::Type> & inferedReturnType
    ) {
      // adaptor yes “incomplete op”，It means only the input is known，Don't know the return value Op
      auto type = adaptor.getValueAttr().getType();
      inferedReturnType.push_back(type);
      return mlir::success();
    }
    ```

###  7.8. <a name='function：functionoptrait'></a>function：FunctionOpTrait

Here is a highlight func、call、return，reference `ex4-beautiful-dialect`。This set of code for the function is very fixed，It's good every time you copy it，There is no much explanation。Set up the function as followsOpback，It should be available `./ex4-opt ../ex4-beautiful-dialect/ex4.mlir` Come to read the function。

####  7.8.1. <a name='definition-return'></a>definition Return

Return is a terminator，Need to use `Terminator`。at the same time，We'll add it to it `ReturnLike`。

```tablegen
def ReturnOp : ToyOp<"ret", [Terminator, ReturnLike]> {
  let summary = "return operation";
  let arguments = (ins AnyType:$data);
  let assemblyFormat = "$data attr-dict `:` type($data)";
}
```

####  7.8.2. <a name='definition-function'></a>definition Function

Defining functions requires implementation `FunctionOpInterface`，It depends on `Symbol` and `CallableOpInterface`。
at the same time，Because we defined it Region，It's better to add `RegionKindInterface`，It will automatically check for us Region Is the format correct?。

Our Function It is a global function，Need to add `IsolatedFromAbove` Trait，This means it won't use what it is located in Region Any previous value。In contrast, `for`，That's not `IsolatedFromAbove`，Because it uses the context's value。

Used here `AnyRegion`，MLIR We also provide some other Region Options，like `SizedRegion<1>` Indicates that there is only one basic block `Regin`。

In order to print the function correctly，We need to call MLIR Provided to us parser and printer，Choose here `hasCustomAssemblyFormat=true`。

Then，To achieve each Interface Required functions，like `extraClassDeclaration` The same inside。

```tablegen
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
```

Then in `toy.cpp`，use MLIR Bring your own function interface Come parse and print。

```cpp
#include "mlir/Interfaces/FunctionImplementation.h"
using namespace mlir;

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](auto & builder, auto argTypes, auto results, auto, auto) {
    return builder.getFunctionType(argTypes, results);
  };
  return function_interface_impl::parseFunctionOp(
    parser, result, false, 
    getFunctionTypeAttrName(result.name), buildFuncType, 
    getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name)
  );
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
    p, *this, false, getFunctionTypeAttrName(),
    getArgAttrsAttrName(), getResAttrsAttrName());
}
```

####  7.8.3. <a name='definition-call'></a>definition Call

use CallOpInterface Just do it，Need to write Interface function。

```tablegen
def CallOp : ToyOp<"call", [CallOpInterface]> {
  let summary = "call operation";
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$arg_operands);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$callee `(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)";
  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      setCalleeAttr(callee.get<mlir::SymbolRefAttr>());
    }
  }];
}
```

##  8. <a name='Add to-pass'></a>Add to Pass

The previous section talked about IR Definition、enter、Output、Member functions, etc.。But a compiler only IR Not enough，Need to be there IR Running on Pass。Our section describes how to use it easily tablegen To define Pass。

###  8.1. <a name='pass-Project templates'></a>Pass Project templates

1. `include/ToyPasses.td`：describe Pass document

    ```tablegen
    include "mlir/Pass/PassBase.td"

    def ConvertToyToArith : Pass<"convert-toy-to-arith"> {
      let summary = "Convert Toy To Arith";
      let constructor = "toy::createConvertToyToArithPass()";
    }
    ```

2. `include/CMakeLists.txt`：Add to tablegen

    ```cmake
    set(LLVM_TARGET_DEFINITIONS ToyPasses.td)
    mlir_tablegen(ToyPasses.h.inc -gen-pass-decls)
    add_public_tablegen_target(MLIRToyTransformsIncGen)
    ```

3. `include/ToyPasses.h`：Pass header file
    ```cpp
    namespace toy {
    // Generate definition
    #define GEN_PASS_DECL
    #include "toy/ToyPasses.h.inc"

    // Writing create Function table
    std::unique_ptr<mlir::Pass> createConvertToyToArithPass();

    // Generate registration function
    #define GEN_PASS_REGISTRATION
    #include "toy/ToyPasses.h.inc"
    }
    ```

4. `lib/Transforms/CMakeLists.txt`：Add to library
    ```cmake
    add_mlir_library(
      ToyTransforms
      ConvertToyToArith.cpp
      DEPENDS MLIRToyTransformsIncGen
    )
    ```

5. `lib/Transforms/ConvertToyToArith.cpp`：A basic implementation
    ```cpp
    #define GEN_PASS_DEF_CONVERTTOYTOARITH
    #include "toy/ToyPasses.h"
    #include "llvm/Support/raw_ostream.h"

    struct ConvertToyToArithPass : 
        toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>
    {
      // Using the constructor of the parent class
      using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;
      void runOnOperation() final {
        getOperation()->print(llvm::errs());
      }
    };

    std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass() {
      return std::make_unique<ConvertToyToArithPass>();
    }
    ```

6. register Pass：`tools/toy-opt/toy-opt.cpp`

    ```cpp
    toy::registerPasses();
    return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
    ```

7. test：`./ex5-opt -convert-toy-to-arith ../ex5-pass/ex5.mlir`

###  8.2. <a name='pass-Definition explanation'></a>Pass Definition explanation

####  8.2.1. <a name='designation-pass-Which one is-op-Run on'></a>designation Pass Which one is Op Run on

reference [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)，MLIR of Pass There are several types below：

* OperationPass：In a fixed type Op Running on Pass
* InterfacePass：In specific OpInterface Running on Pass

The difference between them is，Pass in `getOperation()` The return is Operation still Interface。

By default，`Pass<"convert-toy-to-arith">` Define one at any time `Operation*` Can run on it Pass，If you want to define it in the specified Operation Running on Pass，The following definitions can be used。Notice，Used at this time`toy::FuncOp`，exist `ToyPasses.h` Need in the file include Corresponding header file，Prevent the name from not finding。

```tablegen
def ConvertToyToArith : Pass<"convert-toy-to-arith", "toy::FuncOp"> {
  let summary = "Convert Toy To Arith";
  let constructor = "toy::createConvertToyToArithPass()";
}
```

####  8.2.2. <a name='With parameters-pass'></a>With parameters Pass

First of all, tablgen Write the definition of parameters in the file：

```tablegen
def IsolateTopModule : Pass<"convert-toy-to-arith"> {
  let summary = "Convert Toy To Arith";
  let constructor = "toy::createConvertToyToArithPass()";
  let options = [
    // Name in the code Command line name type    default value   help
    Option<"name", "name", "std::string", "", "help">
  ];
}
```

Then，exist `ToyPasses.h` In the file，To modify create Function definition：

```cpp
std::unique_ptr<mlir::Pass> createConvertToyToArithPass(
  ConvertToyToArithOptions options={}
);
```

In realizing create When the function，Also bring parameters：
```cpp
struct ConvertToyToArithPass : 
    toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>
{
  // Using the constructor of the parent class
  using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;
  void runOnOperation() final {
    llvm::errs() << "get name: " << name << "\n";
  }
};

std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass(
  ConvertToyToArithOptions options) {
  return std::make_unique<ConvertToyToArithPass>(options);
}
```

How to configure parameters：`ex5-opt -convert-toy-to-arith="name=xxx" ../ex5-pass/ex5.mlir`

###  8.3. <a name='Simple-dce-pass-accomplish'></a>Simple DCE Pass accomplish

Pass Implementation，Just use it flexibly IR Traversal and modification。Let's make a simple DCE Pass As Example

####  8.3.1. <a name='definition'></a>definition

```tablegen
def DCE : Pass<"toy-dce", "toy::FuncOp"> {
  let summary = "dce";
  let constructor = "toy::createDCEPass()";
}
```

####  8.3.2. <a name='accomplish'></a>accomplish

```cpp
struct DCEPass : toy::impl::DCEBase<DCEPass> {
  void visitAll(llvm::DenseSet<Operation*> &visited, Operation * op) {
    if(visited.contains(op)) return;
    visited.insert(op);
    for(auto operand: op->getOperands()) 
      if(auto def = operand.getDefiningOp()) 
        visitAll(visited, def);
  }
  void runOnOperation() final {
    llvm::DenseSet<Operation*> visited;
    // Iterate through all Return，Bundle Return Reachable join visited gather
    getOperation()->walk([&](toy::ReturnOp op) {
      visitAll(visited, op);
    });
    llvm::SmallVector<Operation*> opToRemove;
    // Add unreachable opToRemove gather
    getOperation().walk([&](Operation * op) {
      if(op == getOperation()) return;
      if(!visited.contains(op)) opToRemove.push_back(op);
    });
    // Reverse erase
    for(auto v: reverse(opToRemove)) {
      v->erase();
    }
  }
};
```

##  9. <a name='pattern-rewrite'></a>Pattern Rewrite

pattern rewrite yes MLIR A major feature of。Pattern Will match IR A sub-picture of，Then change it to the new format。MLIR It will automatically schedule for us pattern，let IR The transformation is simpler。

a lot of IR All operations can be regarded as Pattern Rewrite：
* Arithmetic optimization，like x*2 Optimized to x+x ，It can be regarded as a pattern replacement for expressions
* expression Lowering，Can be regarded as HighLevel Op Replace with LowLevel Op

In this section，We use Pattern Rewrite Come and make it toy Inside Op Convert to Arith Inside Op。

###  9.1. <a name='pattern-rewrite-1'></a>Pattern Rewrite

####  9.1.1. <a name='describe-pattern'></a>describe Pattern

`matchAndRewrite` return success It means that it can match，return failure It means no match。If it can match，Just pass rewriter rewrite。rewriter A complete set of rewriting API。

```c++
struct AddOpPat: OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op, PatternRewriter & rewriter) const {
    auto inputs = to_vector(op.getInputs());
    auto result = inputs[0];
    for(size_t i = 1; i< inputs.size(); i++) {
      result = rewriter.create<arith::AddIOp>(op->getLoc(), result, inputs[i]);
    }
    rewriter.replaceOp(op, ValueRange(result));
    return success();
  }
};
```

####  9.1.2. <a name='Call-pattern'></a>Call Pattern

In use conversion When，First, define `ConversionTarget`，Then configure it `PatternSet`，Last call `applyXXX` Driver function：

```c++
ConversionTarget target(getContext());
target.addLegalDialect<arith::ArithDialect>();
RewritePatternSet patterns(&getContext());
patterns.add<AddOpPat, SubOpPat, ConstantOpPat>(&getContext());
if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
  signalPassFailure();
```

We used it here `partialConversion`，MLIR Support three types Conversion model：
* `partialConversion`：**if Pattern The conversion result is Legal，Then retain the conversion result**。If the input exists IllegalOp or IllegalDialect，Report an error immediately。
* `fullConversion`：It might beIllegalof。**Call Pattern Convert it，Until all Legal until**。
* `greedyPatternRewrite`：**No need to provide Target，Try to modify as many times as possible**。

The first two are commonly used Dialect Lowering Among them。and`geedyPatternRewrie` Very suitable for writing optimization，For example, I can write a shape like `toy.sub %a, %a` Replace with `const 0: i32` of pattern，hope MLIR Optimize it as much as possible。

####  9.1.3. <a name='depedent-dialect-&-linking'></a>Depedent Dialect & Linking

Notice，we will toy dialect Convert to arith dialect，This shows our pass rely arith ，To add dependencies：

```c++
void getDependentDialects(DialectRegistry &registry) const final {
  registry.insert<arith::ArithDialect>();
}
```

at the same time，exist `opt` Register in the program arith，

```c++
registry.insert<toy::ToyDialect, func::FuncDialect, arith::ArithDialect>();
```

and on arith，This is ours Transform rely arith，so arith Should be loaded transform in the link list。

```cmake
add_mlir_library(
  ToyTransforms
  ConvertToyToArith.cpp
  DCE.cpp
  DEPENDS MLIRToyTransformsIncGen
  LINK_LIBS MLIRArithDialect # here
)
```

Use the following command to verify the result：

```
./ex6-opt --convert-toy-to-arith --toy-dce ../ex6-pattern/ex6.mlir
```


##### debug Method

Can be used `--debug` To start the program，The program will print out the detailed conversion process。

```
./ex6-opt --debug --convert-toy-to-arith ../ex6-pattern/ex6.mlir
```

###  9.2. <a name='dialect-convertion-(type-conversion)'></a>Dialect Convertion (Type Conversion)

Dialect Apart from Op Outside，besides Type。In progress Dialect When the transition between，right Type Rewriting is also very important。

MLIR right Type The method of rewriting is to use `TypeConverter` Completed， `TypeConverter` There are three functions：

1. `addConversion`：Add one Type Conversion rules
2. `addTargetMaterialization`：Generate will SourceType Convert to TargetType Code blocks
3. `addSourceMaterialization`：Generate will TargetType Convert back SourceType Code blocks

The most important of these three is 1，The remaining two do not usually need to be implemented by yourself。

To make a demonstration，We define one's own `toy.int` type，It can be converted to `Integer` type。Skip the type definition part here，Please see more [Custom Type](#10-Customize-type)。

####  9.2.1. <a name='typeconverter'></a>TypeConverter

first，We want to declare one DialectConverter，Then we want to add a type conversion rule to it。The following code has been added ToyIntegerType arrive IntegerType Conversion。MLIR How to use magic template metaprogramming，Get the parameters and return value types of the incoming function，To determine what type to what type of conversion。

```c++
TypeConverter converter;
converter.addConversion([&](ToyIntegerType t) -> std::optional<IntegerType> {
  return IntegerType::get(&getContext(), t.getWidth());
});
```

####  9.2.2. <a name='conversion-pattern：Do it automatically-operand-Type conversion'></a>Conversion Pattern：Do it automatically Operand Type conversion

We use ConversionPattern Automatically convert type。ConversionPattern and RewritePattern The difference is，It has one more `Adaptor`。`Adaptor` in front of [InferTypeOpInterface](#772-Type inferenceinfertypeopinterface) Introduction，`Adaptor` Yes only operands No results The intermediate state of。

MLIR Calling ConversionPattern Before，Will try first op of Operand Convert all to target format，If it cannot be converted, keep the original one。And the converted operand Stored in Adaptor in。

exist replace When，The type can be different from the original one（But it must be able to convert），MLIR It will automatically handle type conversion issues。

```c++
struct AddOpPat: OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AddOp op, AddOpAdaptor adaptor, ConversionPatternRewriter & rewriter) const {
    auto inputs = to_vector(adaptor.getInputs());
    auto result = inputs[0];
    for(size_t i = 1; i< inputs.size(); i++) {
      assert(inputs[i]);
      result = rewriter.create<arith::AddIOp>(op->getLoc(), result, inputs[i]);
    }
    rewriter.replaceOp(op, ValueRange(result));
    return success();
  }
};
```

##### use MLIR Bring your own FuncOpConversion

When performing type conversion to a function，Need to be correct Region Convert the parameter table，You also need to convert the function type。MLIRProvides us with a default Pattern：

```c++
populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);
```

####  9.2.3. <a name='Details of type conversion and-debug'></a>Details of type conversion and Debug

When using type conversion，Can be used `--debug` To start the program，The program will print out the detailed conversion process。

Under normal circumstances，mlir Type conversion is done through a special Op To achieve: `builtin.unrealized_conversion_cast`

For example，mlir To type conversion to the following code：

```mlir
%a = toy.constant 0: !toy.int<32>
%b = toy.add %a, %a: !toy.int<32>
```

certainly，mlir Usually it's replaced from front to back，This is less easy to insert unrealized。Can be apply When configuration config Change order，For display，We assume mlir from%bStart matching。
`toy.add`，Its input is `%a`，Type is `!toy.int<32>`，Convert to `i32`，Need to do TargetMaterialization，Since the user has not registered materialization，Just insert one unrealized Conversion：

```mlir
%a = toy.constant 0: !toy.int<32>
%a_1 = builtin.unrealized_conversion_cast %a : !toy.int<32> to i32
%b = arith.add %a_1, %a_1 : i32
```

Next，Match again `toy.constant`，Replaced for `arith.constant`，mlir I found that the output type had changed，Do SourceMaterialization，Also inserted unrealized

```mlir
%a_2 = arith.constant 0: i32
%a = builtin.unrealized_conversion_cast %a_2 : i32 to !toy.int<32>
%a_1 = builtin.unrealized_conversion_cast %a : !toy.int<32> to i32
%b = arith.add %a_1, %a_1 : i32
```

at last，mlir Will try to put all unrealized Conversion removal。above `i32` Converted to `!toy.int<32>`，Been converted back again，It's invalid conversion，Need to be removed，Finally become：

```mlir
%a = arith.constant 0: i32
%b = arith.add %a, %a : i32
```

####  9.2.4. <a name='Use your own-materialization'></a>Use your own materialization

If the user has registered his own materialization method，MLIR Will register with the user materilzation。

One uses it materialization Scene of：We defined it ourselves `float32` Plural Types，When converting it to `float8`。We definitely want to call functions as little as possible for type conversion `float8`。But if`float32`It's a function parameter，Others cannot change this function if they want to call it，I can only force the conversion。

For example，We can just put `unrealized_conversion_cast` Register as default materialization，This way debug Very convenient when。

```c++
converter.addTargetMaterialization([](OpBuilder& builder, Type /* For all SourceType register */ resultType, ValueRange inputs, Location loc) -> std::optional<Value> {
  return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
});
```

Put about the function target Comment out，You can observe the results of the program：

```c++
// target.addDynamicallyLegalOp<FuncOp>([](FuncOp f) {
//   return llvm::all_of(f.getArgumentTypes(), 
//      [](Type t) {return !isa<ToyIntegerType>(t);});
// });
```

```c++
./ex7-opt --convert-toy-to-arith ../ex7-convert/ex7.mlir
```

```mlir
toy.func @add(%arg0: !toy.int<32>, %arg1: !toy.int<32>) -> !toy.int<32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !toy.int<32> to i32
  %1 = builtin.unrealized_conversion_cast %arg1 : !toy.int<32> to i32
  %2 = arith.addi %0, %1 : i32
  toy.ret %2 : i32
}
```

###  9.3. <a name='use-mlir-Already in it-pattern-Do multi-step conversion'></a>use MLIR Already in it Pattern Do multi-step conversion

MLIR Provides us with modular PatternRewrite API。Almost all Conversion All have corresponding populateXXXPatterns function。

For example，We want to be one-time `toy` Convert to `llvm`，You can write it yourself first pattern Bundle `toy` Convert to `arith`，Add again `arith` of pattern Convert it to `LLVM`：

```c++
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

ConversionTarget target(getContext());
target.addLegalDialect<LLVM::LLVMDialect>();
LLVMTypeConverter converter(&getContext());
RewritePatternSet patterns(&getContext());
patterns.add<AddOpPat, SubOpPat, ConstantOpPat>(&getContext());
arith::populateArithToLLVMConversionPatterns(converter, patterns); // Used pattern
if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
  signalPassFailure();
```

Other header files，The library file that needs to be connected，Please see `ex6` The code in。

##  10. <a name='Customize-type'></a>Customize Type

reference `ex7`，Methods to customize types：

```tablegen
// ToyDialect.td
def ToyDialect : Dialect {
  let name = "toy";
  let cppNamespace = "::toy";
  let summary = "Toy Dialect";
  let useDefaultTypePrinterParser = true; // New
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}


// ToyTypes.td
class ToyType<string name, list<Trait> traits=[]>: TypeDef<ToyDialect, name, traits>;
def ToyInteger: ToyType<"ToyInteger"> {
  let mnemonic = "int";
  let parameters = (ins "uint64_t":$width);
  let assemblyFormat = "`<` $width `>`";
}
```

```c++
#include "toy/ToyTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#define GET_TYPEDEF_CLASSES
#include "toy/ToyTypes.cpp.inc"

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Toy.cpp.inc"
  >();
  registerTypes(); // Increase
}

void ToyDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "toy/ToyTypes.cpp.inc"
  >();
}
```

##  11. <a name='tips'></a>TIPS

###  11.1. <a name='How to find header files、Find the function you want'></a>How to find header files、Find the function you want

first，For commonly used header files，You can check out the function list，include：

* `llvm/ADT/*` The data structure inside
* `mlir/IR/CommonAttrConstraints.td`
* `mlir/IR/CommonTypeConstraints.td`

MLIR of Dialect The file structure is relatively neat，`mlir/Dialect/XXX/IR/XXX.h`

Other functions/Header file，It is recommended to open one vscode arrive mlir Source Code Directory，Use global search to find。

###  11.2. <a name='How to find the library you need to connect to'></a>How to find the library you need to connect to

first，Found you include header file，like `mlir/Dialect/Func/IR/FuncOps.h`。

Then，Find the corresponding header file cpp document，`lib/Dialect/Func/IR/FuncOps.cpp`。

from cpp Search for the file step by step `CMakeLists.txt`，Check the inside `add_mlir_dialect_library` The library file name。

###  11.3. <a name='How to speed up compilation'></a>How to speed up compilation

MLIR It often connects hundreds M Even G Files，Different linkers have a great impact on performance，use `lld` (llvm Linker) It seems to be better `ld` Very fast，The following command can make CMAKE Forced use lld（You need to install it first llvm Compilation toolkit）。

```bash
cmake .. -DCMAKE_CXX_FLAGS="-fuse-ld=lld"
```

###  11.4. <a name='go-mlir-Copy code in'></a>go MLIR Copy code in

MLIR Written a lot for us Dialect，The features we want，Those ones dialect Most of them have been achieved。

Can be used `mlir-opt --help`，`mlir-opt --help-hidden` See what are there dialect What options，Find something that might be similar to what you want to do，Then go and read the code，You can probably do it by reading and copying。

##  12. <a name='mlir-Criticism：c++-v.s.-rust'></a>MLIR Criticism：C++ v.s. Rust

> This is my personal thought，May be more extreme。

MLIR Only use g++ Compilation，use clang++ Compilation meeting Runtime Error。This fully illustrates a fact：MLIR This huge building，It is based on fragility ub (undefined behavior) Above。

MLIR Context，exist rust In the eyes，Actually it's one Interner，Convert type to pointer，Comparison of the coming and going and acceleration types。But in order to achieve it，MLIR Used with obscure and troublesome Pointer, Storage The mode。

Tablegen A paragraph can be described quickly IR，Prerequisite is to remove debug The time required。

Multiple return values Op Although it has increased IR Expression，But make data flow analysis complicated。In fact，Multiple return values Op Almost nonexistent。But I still have to do the rare part Op Make a special judgment，Handle multiple return values。

in addition，As C++ Common problems，MLIR The pointer is so chaotic that it makes people desperate。What makes people even more desperate is，MLIR Requires any operation，IR Always legal。We can't insert a null pointer，You can't delete a variable at will。It can be said，When you walk out PatternRewrite Comfort area，Want to do something complicated Inline, Group, Partition During operation，Segmentation Fault Always inseparable from you。

mlir Innovatively Op Isomorphically viewed as operand, attribute, result Collection of，Specific Op Just the explanation of this set。But its essence，It is a customized serialization and deserialization system。And what is confused is，Such an input and output system，Always exist during operation，We're deserializing anytime Op Come and get it operand，renew operand Then serialize to the general representation。In order to complete such serialization、Deserialization，mlir Create amazing redundant code，Huge binary files，Confusing function definitions，Everywhere Segmentation Fault trap，and unknown performance improvements。

I think，A better one IR System It should be：

* strict SSA of：All operations have return values，Multiple return values ​​are considered Tuple，No return value is considered empty Tuple
* HeterogeneousOp：Op Does not exist operand, attr, result The unified form of，It's heterogeneous，Serialization is done on demand。
* Stateless：unnecessary Interner。Interner It is to handle large amounts of replication。use Rc To handle copying，Implement special Pass Come and go heavy。
* Control flow、Data flow separation：Control flow and data flow are stored in different structures，Can perform separation analysis，Instead of having a pointer table

##  13. <a name='issue-&-reply'></a>Issue & Reply

This document is for teaching only，I am not responsible for the use of mlir Any problems encountered in。As a to use mlir The person，Metaphysics should be done well bug Awareness。
