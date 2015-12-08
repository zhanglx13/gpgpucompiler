package ece.ncsu.edu.gpucompiler.cuda.util;

import java.util.Hashtable;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;

public class ExpressionEval {

	
	public static Expression eval(Expression ex, Hashtable<String, Long> values) throws UnsupportedCodeException {
		if (ex instanceof IntegerLiteral) {
			return ((IntegerLiteral) ex.clone());
		}
		else
		if (ex instanceof Identifier) {
			String name = ((Identifier) ex).getName();
			if (values.get(name)!=null) {
				return new IntegerLiteral(values.get(name));
			}
			else {		
				return (Identifier)ex.clone();
			}
		}		
		else
		if (ex instanceof FunctionCall) {
			throw new UnsupportedCodeException("unknow FunctionCall: "+ex);
		}
		else
		if (ex instanceof ArrayAccess) {
			throw new UnsupportedCodeException("unknow ArrayAccess: "+ex);
		}
		else
		if (ex instanceof BinaryExpression) {
			BinaryExpression be = (BinaryExpression)ex;

			long value = 0;
			Expression lex = eval(be.getLHS(), values);
			Expression rex = eval(be.getRHS(), values);
			if (lex instanceof IntegerLiteral && rex instanceof IntegerLiteral) {
				
				long lhs = ((IntegerLiteral)lex).getValue();
				long rhs = ((IntegerLiteral)rex).getValue();
				if (be.getOperator().equals(BinaryOperator.ADD)||be.getOperator().equals(AssignmentOperator.ADD)) {
					value = lhs + rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.BITWISE_AND )||be.getOperator().equals(AssignmentOperator.BITWISE_AND)) {
					value = lhs & rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.BITWISE_EXCLUSIVE_OR )||be.getOperator().equals(AssignmentOperator.BITWISE_EXCLUSIVE_OR)) {
					value = lhs ^ rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.BITWISE_INCLUSIVE_OR  )||be.getOperator().equals(AssignmentOperator.BITWISE_INCLUSIVE_OR)) {
					value = lhs | rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.DIVIDE )||be.getOperator().equals(AssignmentOperator.DIVIDE)) {
					value = lhs / rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.MODULUS  )||be.getOperator().equals(AssignmentOperator.MODULUS)) {
					value = lhs % rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.MULTIPLY   )||be.getOperator().equals(AssignmentOperator.MULTIPLY)) {
					value = lhs * rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.SHIFT_LEFT  )||be.getOperator().equals(AssignmentOperator.SHIFT_LEFT)) {
					value = lhs << rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.SHIFT_RIGHT  )||be.getOperator().equals(AssignmentOperator.SHIFT_RIGHT)) {
					value = lhs >> rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.SUBTRACT)||be.getOperator().equals(AssignmentOperator.SUBTRACT)) {
					value = lhs - rhs;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.COMPARE_EQ    )) {
					value = lhs == rhs? 1:0;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.COMPARE_GE   )) {
					value = lhs >= rhs? 1:0;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.COMPARE_GT   )) {
					value = lhs > rhs? 1:0;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.COMPARE_LE )) {
					value = lhs <= rhs? 1:0;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.COMPARE_LT    )) {
					value = lhs < rhs? 1:0;
				}
				else 
				if (be.getOperator().equals(BinaryOperator.COMPARE_NE  )) {
					value = lhs != rhs? 1:0;
				}
				else
				if (be.getOperator().equals(AssignmentOperator.NORMAL)) {
					value = rhs;
				}
				else {
					throw new UnsupportedCodeException("unknow BinaryOperation: "+ex);
				}
				
				return new IntegerLiteral(value);
			}
			else
			{
				be = (BinaryExpression)be.clone();
				be.setLHS(lex);
				be.setRHS(rex);
				return be;
			}
		}

		else {
			throw new UnsupportedCodeException("unknow Expression: "+ex);		
		}
	}

}
