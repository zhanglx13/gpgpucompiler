package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import cetus.hir.ArrayAccess;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArray;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class Array2FunctionPass extends Pass {

	boolean isA2F = true;

	public Array2FunctionPass(boolean isA2F) {
		this.isA2F = isA2F;
	}

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}

	@Override
	public void dopass(GProcedure proc) {
		if (isA2F) {
			transformProcedure(proc);
		}
		else {
			transBackProcedure(proc);			
		}
	}

	public void transformArray(ArrayAccess aa) {
		if (aa.getNumIndices() == 1)
			return;
		FunctionCall fc = new FunctionCall((Expression) aa.getArrayName()
				.clone());
		for (int i = 0; i < aa.getNumIndices(); i++)
			fc.addArgument((Expression) aa.getIndex(i).clone());
		aa.swapWith(fc);
	}

	public void transBackArray(FunctionCall fc) {
		// if fc is not array return;
		List indices = new ArrayList();
		for (Object index : fc.getArguments()) {
			Expression ex = (Expression) index;
			indices.add(ex.clone());
		}
		ArrayAccess aa = new ArrayAccess((Expression) fc.getName().clone(),
				indices);
		fc.swapWith(aa);
	}

	public void transformProcedure(GProcedure proc) {
//		System.out.println(proc.toString());
		BreadthFirstIterator iter = new BreadthFirstIterator(proc.getProcedure());

		for (;;) {
			ArrayAccess aa = null;

			try {
				aa = (ArrayAccess) iter.next(ArrayAccess.class);
				if (aa.getNumIndices()!=2) continue;
				MemoryArray  ma = proc.getMemoryArray(aa.getArrayName().toString());
				if (ma==null||ma.getMemoryType()!=MemoryArray.MEMORY_GLOBAL) {
					continue;
				}				
			} catch (NoSuchElementException e) {
				break;
			}

			transformArray(aa);
		}
//		System.out.println(proc.toString());
	}

	public void transBackProcedure(GProcedure proc) {
//		System.out.println(proc.toString());
		BreadthFirstIterator iter = new BreadthFirstIterator(proc.getProcedure());

		for (;;) {
			FunctionCall fc = null;

			try {
				fc = (FunctionCall) iter.next(FunctionCall.class);
				if (fc.getNumArguments()!=2) continue;
				MemoryArray  ma = proc.getMemoryArray(fc.getName().toString());
				if (ma==null||ma.getMemoryType()!=MemoryArray.MEMORY_GLOBAL) {
					continue;
				}
			} catch (NoSuchElementException e) {
				break;
			}

			transBackArray(fc);
		}
//		System.out.println(proc.toString());
	}

}
