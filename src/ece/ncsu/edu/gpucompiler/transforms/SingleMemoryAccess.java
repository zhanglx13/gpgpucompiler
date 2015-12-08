package ece.ncsu.edu.gpucompiler.transforms;

import java.util.NoSuchElementException;

import cetus.hir.BreadthFirstIterator;
import cetus.hir.ExpressionStatement;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;

/**
 * one statement only access one memory
 * for example
 * input:
 * c[i] += a[i]*b[i];
 * output:
 * a0 = a[i];
 * b0 = b[i];
 * c[i] += a0*b0; 
 * @author jack
 *
 */
public class SingleMemoryAccess {

	public void transformProcedure(GProcedure proc) {
	    BreadthFirstIterator iter = new BreadthFirstIterator(proc.getProcedure());

	    for (;;)
	    {
	      ExpressionStatement stmt = null;

	      try {
	    	  stmt = (ExpressionStatement)iter.next(ExpressionStatement.class);
	      } catch (NoSuchElementException e) {
	        break;
	      }

	      transformStatement(stmt, proc);
	    }
	}

	private void transformStatement(ExpressionStatement stmt, GProcedure proc) {

		
	}
	
	
}
