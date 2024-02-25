use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicType, BasicTypeEnum, FunctionType};
use inkwell::values::{BasicValue, BasicValueEnum, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Binary;
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
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Lte,
    Gt,
    Gte,
    And,
    Or,
    
}

#[derive(Debug, Clone, PartialEq)]
struct BinaryExpr {
    op: BinaryOp,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Literal(Literal),
    Identifier(Identifier),
    BinaryExpr(BinaryExpr),
    
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

#[derive(Debug, Clone, PartialEq)]
struct Symbol<'ctx> {
    pointer: PointerValue<'ctx>,
    pointee_type: BasicTypeEnum<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn codegen_expr(&mut self, symbol_table: &mut HashMap<Identifier, Symbol<'ctx>>, expr: &Expr) -> Result<Option<BasicValueEnum<'ctx>>, Box<dyn Error>> {
        match expr {

            Expr::FuncDef(func_def) => {
     
                let param_types = vec![];
                let func = match func_def.return_type {
                    TypeExpr::Int => self.context.i64_type().fn_type(&param_types, false),
                    TypeExpr::Float => self.context.f64_type().fn_type(&param_types, false),
                    TypeExpr::String => self.context.i8_type().ptr_type(AddressSpace::default()).fn_type(&param_types, false),
                    TypeExpr::Bool => self.context.bool_type().fn_type(&param_types, false),
                    TypeExpr::Void => self.context.void_type().fn_type(&param_types, false),                       
                    _ => unimplemented!()
                }; 

                let function_name = func_def.name.as_str();

                let function = self.module.add_function(function_name, func, None);

                let basic_block = self.context.append_basic_block(function, "entry");

                self.builder.position_at_end(basic_block);

                // generate the function body

                let body = self.codegen_expr(symbol_table, &*func_def.body)?; 


                Ok(None)
            },
            Expr::BinaryExpr(binary)  => {
                let lhs_val = self.codegen_expr(symbol_table, &binary.lhs)?.unwrap();
                let rhs_val = self.codegen_expr(symbol_table, &binary.rhs)?.unwrap();

                // check if the types are the same
                if lhs_val.get_type() != rhs_val.get_type() {
                    return Err("Types do not match".into());
                }

                    //    match type int, float 



                if(lhs_val.is_int_value()){ 
                    let result = match binary.op {
                        BinaryOp::Add => self.builder.build_int_add(lhs_val.into_int_value(), rhs_val.into_int_value(), "addtmp"),
                        BinaryOp::Sub => self.builder.build_int_sub(lhs_val.into_int_value(), rhs_val.into_int_value(), "subtmp"),
                        BinaryOp::Mul => self.builder.build_int_mul(lhs_val.into_int_value(), rhs_val.into_int_value(), "multmp"),
                        BinaryOp::Div => self.builder.build_int_signed_div(lhs_val.into_int_value(), rhs_val.into_int_value(), "divtmp"),
                        BinaryOp::Mod => self.builder.build_int_signed_rem(lhs_val.into_int_value(), rhs_val.into_int_value(), "modtmp"),
                        BinaryOp::Eq => self.builder.build_int_compare(inkwell::IntPredicate::EQ, lhs_val.into_int_value(), rhs_val.into_int_value(), "eqtmp"),
                        BinaryOp::Ne => self.builder.build_int_compare(inkwell::IntPredicate::NE, lhs_val.into_int_value(), rhs_val.into_int_value(), "netmp"),
                        BinaryOp::Lt => self.builder.build_int_compare(inkwell::IntPredicate::SLT, lhs_val.into_int_value(), rhs_val.into_int_value(), "lttmp"),
                        BinaryOp::Lte => self.builder.build_int_compare(inkwell::IntPredicate::SLE, lhs_val.into_int_value(), rhs_val.into_int_value(), "ltetmp"),
                        BinaryOp::Gt => self.builder.build_int_compare(inkwell::IntPredicate::SGT, lhs_val.into_int_value(), rhs_val.into_int_value(), "gttmp"),
                        BinaryOp::Gte => self.builder.build_int_compare(inkwell::IntPredicate::SGE, lhs_val.into_int_value(), rhs_val.into_int_value(), "gtetmp"),
                        BinaryOp::And => self.builder.build_and(lhs_val.into_int_value(), rhs_val.into_int_value(), "andtmp"),
                        BinaryOp::Or => self.builder.build_or(lhs_val.into_int_value(), rhs_val.into_int_value(), "ortmp"),
                    };
                    return Ok(Some(result.unwrap().into()));
                }
                // if (lhs_val.is_float_value()) {
                //     let result = match binary.op {
                //         BinaryOp::Add => self.builder.build_float_add(lhs_val.into_float_value(), rhs_val.into_float_value(), "addtmp"),
                //         BinaryOp::Sub => self.builder.build_float_sub(lhs_val.into_float_value(), rhs_val.into_float_value(), "subtmp"),
                //         BinaryOp::Mul => self.builder.build_float_mul(lhs_val.into_float_value(), rhs_val.into_float_value(), "multmp"),
                //         BinaryOp::Div => self.builder.build_float_div(lhs_val.into_float_value(), rhs_val.into_float_value(), "divtmp"),
                //         BinaryOp::Mod => self.builder.build_float_rem(lhs_val.into_float_value(), rhs_val.into_float_value(), "modtmp"),
                //         // BinaryOp::Eq => self.builder.build_float_compare(inkwell::FloatPredicate::OEQ, lhs_val.into_float_value(), rhs_val.into_float_value(), "eqtmp"),
                //         // BinaryOp::Ne => self.builder.build_float_compare(inkwell::FloatPredicate::ONE, lhs_val.into_float_value(), rhs_val.into_float_value(), "netmp"),
                //         // BinaryOp::Lt => self.builder.build_float_compare(inkwell::FloatPredicate::OLT, lhs_val.into_float_value(), rhs_val.into_float_value(), "lttmp"),
                //         // BinaryOp::Lte => self.builder.build_float_compare(inkwell::FloatPredicate::OLE, lhs_val.into_float_value(), rhs_val.into_float_value(), "ltetmp"),
                //         // BinaryOp::Gt => self.builder.build_float_compare(inkwell::FloatPredicate::OGT, lhs_val.into_float_value(), rhs_val.into_float_value(), "gttmp"),
                //         // BinaryOp::Gte => self.builder.build_float_compare(inkwell::FloatPredicate::OGE, lhs_val.into_float_value(), rhs_val.into_float_value(), "gtetmp"),
                //         // BinaryOp::And => self.builder.build_and(lhs_val.into_int_value(), rhs_val.into_int_value(), "andtmp"),
                //         // BinaryOp::Or => self.builder.build_or(lhs_val.into_int_value(), rhs_val.into_int_value(), "ortmp"),
                //         _ => unimplemented!()
                //     }?;
                // }
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
                    let symbol = match symbol_table.get(name) {
                        Some(sym) => sym,
                        None => return Err("Variable not found".into()),
                    };

                    let pointer_value = symbol.pointer;
                    let pointee_type = symbol.pointee_type;

                    let value = self
                        .builder
                        .build_load(pointee_type, pointer_value, name)?;
                    // dump the value
                    Ok(Some(value))

                } else {
                    Err("Variable not found".into())
                }
            }
            Expr::IfExpr(IfExpr) => {
                let condition = self.codegen_expr(symbol_table, &IfExpr.condition)?.unwrap();
                let condition = self.builder.build_int_compare(inkwell::IntPredicate::NE, condition.into_int_value(), self.context.i64_type().const_zero(), "ifcond")?;
                let function = self.builder.get_insert_block().unwrap().get_parent().unwrap();
                let then_block = self.context.append_basic_block(function, "then");
                let else_block = self.context.append_basic_block(function, "else");
                let merge_block = self.context.append_basic_block(function, "ifcont");

                self.builder.build_conditional_branch(condition, then_block, else_block);

                self.builder.position_at_end(then_block);
                let then_val = self.codegen_expr(symbol_table, &IfExpr.then_expr)?.unwrap();
                self.builder.build_unconditional_branch(merge_block);
                let then_block = self.builder.get_insert_block().unwrap();

                self.builder.position_at_end(else_block);
                let else_val = self.codegen_expr(symbol_table, &IfExpr.else_expr)?.unwrap();
                self.builder.build_unconditional_branch(merge_block);
                let else_block = self.builder.get_insert_block().unwrap();

                self.builder.position_at_end(merge_block);
                let phi = self.builder.build_phi(self.context.i64_type(), "iftmp")?;
                phi.add_incoming(&[(&then_val, then_block), (&else_val, else_block)]);
                Ok(Some(phi.as_basic_value()))
            }

            Expr::LoopExpr(loop_expr) => {
                let loop_block = self.context.append_basic_block(self.builder.get_insert_block().unwrap().get_parent().unwrap(), "loop");
                let after_block = self.context.append_basic_block(self.builder.get_insert_block().unwrap().get_parent().unwrap(), "afterloop");

                self.builder.build_unconditional_branch(loop_block);
                self.builder.position_at_end(loop_block);

                let condition = self.codegen_expr(symbol_table, &loop_expr.condition)?.unwrap();
                let condition = self.builder.build_int_compare(inkwell::IntPredicate::NE, condition.into_int_value(), self.context.i64_type().const_zero(), "loopcond")?;
                self.builder.build_conditional_branch(condition, loop_block, after_block);

                self.builder.position_at_end(after_block);

                Ok(None)
            }
            Expr::Block(block) => {
                let mut temp_table = symbol_table.clone();
                for expr in &block.exprs {
                    self.codegen_expr(&mut temp_table, expr)?;
                }
                Ok(None)
            }
           
            Expr::Assignment(assignment) => {
                let identifier = &assignment.name;
                let rhs_val = self.codegen_expr(symbol_table, &assignment.value)?.unwrap();

                 let val: Result<Option<BasicValueEnum<'_>>, Box<dyn Error>> = match symbol_table.get(identifier) {
                    Some(symbol) => {
                        let ptr = symbol.pointer;
                        let res = self.builder.build_store(ptr, rhs_val)?;
                        Ok(None)
                    }
                    None => {
                        let ptr = self.builder.build_alloca(rhs_val.get_type(), identifier)?;

                        let res = self.builder.build_store(ptr, rhs_val)?;
                        symbol_table.insert(identifier.to_string(), Symbol { pointer: ptr, pointee_type: rhs_val.get_type() }); 
                        Ok(None)
                    }
                };
                val
            }
            Expr::Return(expr) => {
   
                let return_val = self.codegen_expr(symbol_table, expr)?.unwrap();


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
    let mut symbol_table = HashMap::<Identifier, Symbol>::new();
    let main_ast = Expr::FuncDef(FuncDef {
        name: "main".to_string(),
        return_type: TypeExpr::Int,
        args: vec![],
        body: Box::new(Expr::Block(
            Block {
                exprs: vec![
                    Expr::Assignment(Assignment {
                        name: "x".to_string(),
                        value: Box::new(Expr::Literal(Literal::Int(42))),
                    }),
                    // loop(x < 100) { x = x + 1 }
                    Expr::LoopExpr(LoopExpr {
                        condition: Box::new(Expr::BinaryExpr(BinaryExpr {
                            op: BinaryOp::Lt,
                            lhs: Box::new(Expr::Identifier("x".to_string())),
                            rhs: Box::new(Expr::Literal(Literal::Int(100))),
                        })),
                        body: Box::new(Expr::Block(Block {
                            exprs: vec![Expr::Assignment(Assignment {
                                name: "x".to_string(),
                                value: Box::new(Expr::FuncCall {
                                    name: "+".to_string(),
                                    args: vec![
                                        Expr::Identifier("x".to_string()),
                                        Expr::Literal(Literal::Int(1)),
                                    ],
                                }),
                            })],
                        })),
                    }),
                    Expr::Return(Box::new(Expr::Identifier("x".to_string()))),
                ],
            }
        )),
    });

    codegen.codegen_expr(&mut symbol_table, &main_ast)?;

    codegen.dump();

    Ok(())
}