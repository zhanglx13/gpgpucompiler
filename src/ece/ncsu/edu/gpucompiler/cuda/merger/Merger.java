package ece.ncsu.edu.gpucompiler.cuda.merger;

import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;

public abstract class Merger {

	public final static int MERGE_DIRECTION_X = 0;
	public final static int MERGE_DIRECTION_Y = 1;

	public final static int MERGE_TYPE_THREAD = 0;
	public final static int MERGE_TYPE_THREADBLOCK = 1;


	
	MergeAction mergeAction = null;
	

	public MergeAction getMergeAction() {
		return mergeAction;
	}
	public void setMergeAction(MergeAction mergeAction) {
		this.mergeAction = mergeAction;
	}
	
	
	public abstract void merge(GProcedure proc);
	
}
