package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.List;

import cetus.hir.DepthFirstIterator;
import cetus.hir.ForLoop;
import cetus.hir.Loop;
import cetus.hir.Procedure;
import cetus.hir.Traversable;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class GLoopPass extends Pass {

	
	
	public GLoopPass() {
		
	}
	
	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}
	
	@Override
	public void dopass(GProcedure proc) {
		Procedure procedure = proc.getProcedure();
		DepthFirstIterator dfi = new DepthFirstIterator(procedure);
		boolean hasloop = true;
		while (hasloop) {
			hasloop = false;
			dfi = new DepthFirstIterator(procedure);
			List<Loop> loops = (List<Loop>)dfi.getList(ForLoop.class);
			GLoop last = null;
			for (Loop loop: loops) {
				if (loop instanceof GLoop) continue;
				try {
					ForLoop forloop = (ForLoop)loop;
					GLoop gloop = new GLoop(forloop, proc);
					forloop.swapWith(gloop);
	//				StatementUtil.replace(forloop, gloop);
					if (last==null) last = gloop;
					else {
						Traversable tr = gloop;
						while (tr.getParent()!=null) {
							tr = tr.getParent();
							if (tr.equals(last)) {
								gloop.setLoopParent(last);
								break;
							}
		 				}
					}
					last = gloop;
					hasloop = true;
				}
				catch (Exception ex) {
					System.out.println("not gloop:"+ex.getMessage());
//					ex.printStackTrace();
				}
			}
		}
	}

}
