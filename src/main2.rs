use std::collections::HashMap;
use std::ffi::CStr;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr::null;
use std::str;

use llvm_sys::core::*;
use llvm_sys::error::LLVMErrorRef;
use llvm_sys::execution_engine::*;
use llvm_sys::ir_reader::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use llvm_sys::target_machine::*;
use llvm_sys::*;

struct Symbol  {

    llvm_val: ?LLVMValueRef,
}
struct LLVMContext {
    context: LLVMContextRef,
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    symbol_table: HashMap<Identifier, LLVMValueRef>,
}

impl LLVMContext {
    fn initialize() {
        unsafe {
            if LLVM_InitializeNativeTarget() != 0 {
                panic!("Failed to initialize native target")
            }
            if LLVM_InitializeNativeAsmParser() != 0 {
                panic!("Failed to initialize native asm parser")
            } 
            if LLVM_InitializeNativeAsmPrinter() != 0 {
                panic!("Failed to initialize native asm printer")
            } 
            if LLVM_InitializeNativeDisassembler() != 0 {
                panic!("Failed to initialize native disassembler")
            }
        }
    }
    fn new() -> LLVMContext {
        unsafe {
            let context = LLVMContextCreate();
            let builder = LLVMCreateBuilderInContext(context);
            let module = LLVMModuleCreateWithNameInContext("main\0".as_ptr() as *const i8, context);
            let symbol_table = HashMap::new();
            LLVMContext {
                context: context,
                builder: builder,
                module: module,
                symbol_table: symbol_table,
            }
        }
    }

    pub fn codegen_if_statement(&mut self, if_statement: IfStatement) {
        let if_block = unsafe {
            let current_block = LLVMGetInsertBlock(self.builder);
            // check if the current block is not null
            assert!(!current_block.is_null());
            let func = LLVMGetBasicBlockParent(current_block);
            // check if the parent function is not null
            assert!(!func.is_null());
            // Convert LLVMGetBasicBlockName result to Rust String
            let c_str_current_block_name = CStr::from_ptr(LLVMGetBasicBlockName(current_block));
            let current_block_name =
                str::from_utf8(c_str_current_block_name.to_bytes()).expect("Invalid UTF-8 data");

            // Convert LLVMGetValueName result to Rust String
            let c_str_func_name = CStr::from_ptr(LLVMGetValueName(func));
            let func_name = str::from_utf8(c_str_func_name.to_bytes()).expect("Invalid UTF-8 data");

            // Now format the Rust Strings
            let then_block_name = format!("{}_{}_then_block", func_name, current_block_name);
            let else_block_name = format!("{}_{}_else_block", func_name, current_block_name);
            let merge_block_name = format!("{}_{}_merge_block", func_name, current_block_name);

            let then_block = LLVMAppendBasicBlockInContext(
                self.context,
                func,
                then_block_name.as_ptr() as *const i8,
            );
            let else_block = LLVMAppendBasicBlockInContext(
                self.context,
                func,
                else_block_name.as_ptr() as *const i8,
            );
            let merge_block = LLVMAppendBasicBlockInContext(
                self.context,
                func,
                merge_block_name.as_ptr() as *const i8,
            );

            // codegen the condition
            let condition = self.codegen_expr(*if_statement.condition);

            // LLVMDumpValue(condition);

            let condition_cast = LLVMBuildIntCast(
                self.builder,
                condition,
                LLVMInt1TypeInContext(self.context),
                "condition_cast\0".as_ptr() as *const i8,
            );

            LLVMBuildCondBr(self.builder, condition_cast, then_block, else_block);
            LLVMPositionBuilderAtEnd(self.builder, then_block);
            // codegen the then block
            self.codegen_expr(*if_statement.then_block);
            // check if previous was a return statement
            let then_terminator = LLVMGetBasicBlockTerminator(then_block);
            if then_terminator.is_null() {
                LLVMBuildBr(self.builder, merge_block);
            }

            LLVMPositionBuilderAtEnd(self.builder, else_block);
            // codegen the else block
            self.codegen_expr(*if_statement.else_block);

            let else_terminator = LLVMGetBasicBlockTerminator(else_block);
            if else_terminator.is_null() {
                LLVMBuildBr(self.builder, merge_block);
            }

            LLVMPositionBuilderAtEnd(self.builder, merge_block);

            let phi_node = LLVMBuildPhi(
                self.builder,
                LLVMInt32TypeInContext(self.context),
                "phi\0".as_ptr() as *const i8,
            );
            // add incoming values to phi_node
            let mut incoming_values = vec![
                LLVMConstInt(LLVMInt32TypeInContext(self.context), 1, 0),
                LLVMConstInt(LLVMInt32TypeInContext(self.context), 0, 0),
            ];
            let mut incoming_blocks = vec![then_block, else_block];
            LLVMAddIncoming(
                phi_node,
                incoming_values.as_mut_ptr(),
                incoming_blocks.as_mut_ptr(),
                2,
            );
            phi_node
        };
    }

    pub fn codegen_binary_expr(&mut self, binary_expr: BinaryExpr) -> LLVMValueRef {
        match binary_expr.op {
            BinaryOp::Assign => {
                let lhs = self.codegen_expr(*binary_expr.lhs);
                let rhs = self.codegen_expr(*binary_expr.rhs);
                unsafe {
                    LLVMBuildStore(self.builder, rhs, lhs);
                }
                lhs
            }
            BinaryOp::Equal => {
                let lhs = self.codegen_expr(*binary_expr.lhs);
                let rhs = self.codegen_expr(*binary_expr.rhs);
                unsafe {
                    LLVMBuildICmp(
                        self.builder,
                        LLVMIntPredicate::LLVMIntEQ,
                        lhs,
                        rhs,
                        "equal\0".as_ptr() as *const i8,
                    )
                }
            }
        }
    }
    pub fn codegen_literal(&mut self, literal: Literal) -> LLVMValueRef {
        match literal {
            Literal::Int(int) => unsafe {
                LLVMConstInt(LLVMInt32TypeInContext(self.context), int as u64, 0)
            },
        }
    }

    pub fn codegen_identifier(&mut self, identifier: Identifier) -> LLVMValueRef {
        //   check to see if we have a context block
        let current_block = unsafe { LLVMGetInsertBlock(self.builder) };

        // check if the current block is not null we want a build alloca
        if !current_block.is_null() {
            // check to see if it is defined already in symbol table
            let value = self.symbol_table.get_mut(&identifier);
            match value {
                Some(value) => {
                    return unsafe {
                        let llvm_val = *value;
                        let llvm_ty = LLVMTypeOf(llvm_val);
                        LLVMBuildLoad2(
                            self.builder,
                            llvm_ty,
                            llvm_val,
                            identifier.as_ptr() as *const i8,
                        )
                    }
                }
                None => {
                    return unsafe {
                        let alloca = LLVMBuildAlloca(
                            self.builder,
                            LLVMInt32TypeInContext(self.context),
                            identifier.as_ptr() as *const i8,
                        );
                        LLVMSetAlignment(alloca, 4);
                        self.symbol_table.insert(identifier, alloca);
                        alloca
                    }
                }
            }
        }

        // throw an error if we don't have a context block
        panic!("No context block");
    }

    pub fn codegen_func_def(&mut self, func_def: FuncDef) -> LLVMValueRef {
        let func = unsafe {
            let func_ty = LLVMFunctionType(
                LLVMVoidTypeInContext(self.context),
                std::ptr::null_mut(),
                0,
                0,
            );
            let func = LLVMAddFunction(
                self.module,
                func_def.name.as_ptr() as *const i8,
                func_ty,
            );
            LLVMSetLinkage(func, LLVMLinkage::LLVMExternalLinkage);
            func
        };
        self.symbol_table.insert(func_def.name.clone(), func);
        unsafe {
            let current_block = LLVMAppendBasicBlockInContext(
                self.context,
                func,
                "entry\0".as_ptr() as *const i8,
            );
            LLVMPositionBuilderAtEnd(self.builder, current_block);
        }
        self.codegen_expr(*func_def.body);
        func
    }

    pub fn codegen_stmt(&mut self, stmt: Stmt) -> LLVMValueRef {
        for expr in stmt.exprs {
            match expr {
                Expr::Literal(literal) => {
                    // codegen a return statement with the literal
                    let literal = self.codegen_literal(literal);
                    unsafe {
                        LLVMBuildRet(self.builder, literal);
                    }
                }
                Expr::Identifier(identifier) => {
                    // codegen a return statement with the identifier
                    let identifier = self.codegen_expr(Expr::Identifier(identifier));
                    unsafe {
                        LLVMBuildRet(self.builder, identifier);
                    }
                }
                _ => {
                    self.codegen_expr(expr);
                }
            }
        }
        unsafe { LLVMConstInt(LLVMInt32TypeInContext(self.context), 1, 0) }
    }
    pub fn codegen_expr(&mut self, expr: Expr) -> LLVMValueRef {
        match expr {
            Expr::Literal(literal) =>  return self.codegen_literal(literal),
            Expr::Identifier(identifier) => return self.codegen_identifier(identifier),
            Expr::Stmt(stmt) => return self.codegen_stmt(stmt),
            Expr::IfStatement(if_statement) =>  self.codegen_if_statement(if_statement),
            Expr::BinaryExpr(binary_expr) => return self.codegen_binary_expr(binary_expr),
            Expr::FuncDef(func_def) => return self.codegen_func_def(func_def),
            Expr::ReturnExpr(expr) =>{
                let expr = self.codegen_expr(*expr);
                unsafe {
                    LLVMBuildRet(self.builder, expr);
                }
            }

        }
        unsafe { LLVMConstInt(LLVMInt32TypeInContext(self.context), 1, 0) }
    }
 
    pub fn dump(&self) {
        unsafe {
            LLVMDumpModule(self.module);
        }
    }
}
#[derive(Debug, Clone)]

enum BinaryOp {
    Equal,
    Assign,
}

#[derive(Debug, Clone)]

struct BinaryExpr {
    lhs: Box<Expr>,
    rhs: Box<Expr>,
    op: BinaryOp,
}
#[derive(Debug, Clone)]

enum Literal {
    Int(i64),
}

#[derive(Debug, Clone)]

// stmt = vec<expr>
struct Stmt {
    exprs: Vec<Expr>,
}

type Identifier = String;

#[derive(Debug, Clone)]
struct Type {
    name: Identifier,
}

#[derive(Debug, Clone)]
struct Arg {
    name: Identifier,
    ty: Type,
}

#[derive(Debug, Clone)]
struct FuncDef {
    name: Identifier,
    args: Vec<Arg>,
    return_ty: Type,
    body: Box<Expr>,
}

#[derive(Debug, Clone)]
enum Expr {
    Identifier(Identifier),
    Literal(Literal),
    BinaryExpr(BinaryExpr),
    IfStatement(IfStatement),
    Stmt(Stmt),
    FuncDef(FuncDef),
    ReturnExpr(Box<Expr>),
}
#[derive(Debug, Clone)]

struct IfStatement {
    condition: Box<Expr>,
    then_block: Box<Expr>,
    else_block: Box<Expr>,
}

fn main() {
    // initialize LLVM
    LLVMContext::initialize();
    let mut llvm_ctx = LLVMContext::new();



    let main_func = Expr::FuncDef(FuncDef {
        name: "main".to_string(),
        args: vec![],
        return_ty: Type {
            name: "i32".to_string(),
        },
        body: Box::new(Expr::Stmt(Stmt{
            exprs: vec![
                // a = 1
                Expr::BinaryExpr(BinaryExpr {
                    lhs: Box::new(Expr::Identifier("a".to_string())),
                    rhs: Box::new(Expr::Literal(Literal::Int(1))),
                    op: BinaryOp::Assign,
                }),
                // b = 2
                Expr::BinaryExpr(BinaryExpr {
                    lhs: Box::new(Expr::Identifier("b".to_string())),
                    rhs: Box::new(Expr::Literal(Literal::Int(1))),
                    op: BinaryOp::Assign,
                }),

                // if a == b { return 1 } else { return 0 }
                Expr::IfStatement(IfStatement {
                    condition: Box::new(Expr::BinaryExpr(BinaryExpr {
                        lhs: Box::new(Expr::Identifier("a".to_string())),
                        rhs: Box::new(Expr::Identifier("b".to_string())),
                        op: BinaryOp::Equal,
                    })),
                    then_block: Box::new(Expr::Stmt(Stmt {
                        exprs: vec![Expr::Literal(Literal::Int(1))],
                    })),
                    else_block: Box::new(Expr::Stmt(Stmt {
                        exprs: vec![Expr::Literal(Literal::Int(0))],
                    })),
                }),
            ],
        })),
    });

    llvm_ctx.codegen_expr(main_func);

    llvm_ctx.dump();

    // execution engine


    let execution_engine = unsafe {
        LLVMLinkInMCJIT(); 
        let mut out_ee = MaybeUninit::uninit();
        let mut out_error = MaybeUninit::uninit();

        if LLVMCreateExecutionEngineForModule(out_ee.as_mut_ptr(), llvm_ctx.module, out_error.as_mut_ptr()) != 0 { 
            let error_message = CStr::from_ptr(out_error.assume_init()).to_str().unwrap();
            panic!("Failed to create execution engine: {}", error_message);
        }
        out_ee.assume_init()
    };

   
    let result = unsafe {
        LLVMRunFunctionAsMain(
            execution_engine,
            llvm_ctx.symbol_table.get("main").unwrap().clone(),
            0,
            null(),
            null(),
        )
    };

    print!("Result: {}", result);
}
