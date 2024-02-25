use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::{Linkage, Module};
use inkwell::values::{BasicValue, BasicValueEnum, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

use std::collections::HashMap;
use std::error::Error;
use std::ops::Deref;

type Identifier = String;

#[derive(Debug, Clone, PartialEq)]
enum TypeExpr {
    Int,
    Float,
    String,
    Bool,
    Void,
    FuncDef {
        args: Vec<TypeExpr>,
        return_type: Box<TypeExpr>,
    },
    StructDef {
        name: Identifier,
        fields: Vec<(Identifier, TypeExpr)>,
    }
    
}

#[derive(Debug, Clone, PartialEq)]
struct FuncDef {
    name: Identifier,
    return_type: TypeExpr,
    args: Vec<(Identifier, TypeExpr)>,
    body: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
struct Assignment{
    name: Identifier,
    value: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
struct IfExpr {
    condition: Box<Expr>,
    then_expr: Box<Expr>,
    else_expr: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
struct LoopExpr {
    condition: Box<Expr>,
    body: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
struct Block {
    exprs: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Literal(Literal),
    Identifier(Identifier),
    
    Assignment(Assignment),

    IfExpr(IfExpr),
    LoopExpr(LoopExpr),

    FuncDef(FuncDef),
    FuncCall {
        name: Identifier,
        args: Vec<Expr>,
    },
    Block(Block),
    Return(Box<Expr>),

}

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn codegen_expr( &mut self, mut symbol_table: HashMap<Identifier, PointerValue<'ctx>>, expr: &Expr) -> Result<Option<BasicValueEnum<'ctx>>, Box<dyn Error>> {
        match expr {

            Expr::FuncDef(func_def) => {
     
                let param_types = vec![];
                let func = self.context.i32_type().fn_type(&param_types, false);

                let function_name = func_def.name.as_str();

                let function = self.module.add_function(function_name, func, None);

                let basic_block = self.context.append_basic_block(function, "entry");

                self.builder.position_at_end(basic_block);

                // generate the function body

                let body = self.codegen_expr(symbol_table.clone(), &*func_def.body)?; 


                Ok(None)
            }

            Expr::Literal(literal) => match literal {
                Literal::Int(int) => {
                    let int_type = self.context.i64_type();
                    let int_value = int_type.const_int(*int as u64, false);
                    Ok(Some(int_value.into()))
                }
                Literal::Float(float) => {
                    let float_type = self.context.f64_type();
                    let float_value = float_type.const_float(*float);
                    Ok(Some(float_value.into()))
                }
                Literal::String(string) => {
                    let string_value = self.context.const_string(string.as_bytes(), false);
                    Ok(Some(string_value.into()))
                }
                Literal::Bool(bool) => {
                    let bool_type = self.context.bool_type();
                    let bool_value = bool_type.const_int(*bool as u64, false);
                    Ok(Some(bool_value.into()))
                }
            },
            Expr::Identifier(name) => {
                // try to find the variable in the current scope
                // if not found, try to find the variable in the global scope
                // if not found, return an error

                if (symbol_table.contains_key(name)) {
                    let pointer_value = match symbol_table.get(name) {
                        Some(pointer_value) => pointer_value,
                        None => return Err("Variable not found".into()),
                    };
                    let pointer_type = pointer_value.get_type();

                    let value = self
                        .builder
                        .build_load(pointer_type, *pointer_value, name)?;

                    Ok(Some(value))
                } else {
                    Err("Variable not found".into())
                }
            }

            Expr::Block(block) => {
                for expr in &block.exprs {
                    self.codegen_expr(symbol_table.clone(), expr)?;
                }
                Ok(None)
            }
           
            // In Expr::Assignment
            Expr::Assignment(assignment) => {
                let identifier = &assignment.name;
                let rhs_val = self.codegen_expr(symbol_table.clone(), &assignment.value)?.unwrap();

                 let val = match symbol_table.clone().get(identifier) {
                    Some(val) => {
                        let res = self.builder.build_store(*val, rhs_val)?;
                        Ok(None)
                    }
                    None => {
                        let alloca = self.builder.build_alloca(rhs_val.get_type(), identifier)?;
                        let res = self.builder.build_store(alloca, rhs_val)?;
                        symbol_table.insert(identifier.to_string(), alloca);
                        Ok(None)
                    }
                };
                val
            }
            Expr::Return(expr) => {
                let return_val = self.codegen_expr(symbol_table.clone(), expr)?.unwrap();
                self.builder.build_return(Some(&return_val));
                Ok(None)
            }

            _ => unimplemented!("codegen_expr for {:?}", expr),
        }
    }

    fn dump(&self) {
        self.module.print_to_stderr();
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("main");

    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;

    let mut codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };
    let mut symbol_table = HashMap::new();

    let main_ast = Expr::FuncDef(FuncDef {
        name: "main".to_string(),
        return_type: TypeExpr::Void,
        args: vec![],
        body: Box::new(Expr::Block(
            Block {
                exprs: vec![
                    Expr::Assignment(Assignment {
                        name: "x".to_string(),
                        value: Box::new(Expr::Literal(Literal::Int(42))),
                    }),
                    Expr::Return(Box::new(Expr::Identifier("x".to_string()))),
                ],
            }
        )),
    });

    codegen.codegen_expr(symbol_table, &main_ast)?;

    codegen.dump();

    Ok(())
}
