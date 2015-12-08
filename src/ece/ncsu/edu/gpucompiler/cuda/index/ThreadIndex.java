package ece.ncsu.edu.gpucompiler.cuda.index;

public class ThreadIndex extends Index {

	// local thread size
	public final static String BLOCK_DIM_X = "blockDimX";	// blockDim.x
	public final static String BLOCK_DIM_Y = "blockDimY";
	
	// global block size
	public final static String GRID_DIM_X = "gridDimX";
	public final static String GRID_DIM_Y = "gridDimY";

	// global thread id
	public final static String IDX = "idx";
	public final static String IDY = "idy";
	
	// block id
	public final static String BIDX = "bidx";
	public final static String BIDY = "bidy";

	// local thread id
	public final static String TIDX = "tidx";
	public final static String TIDY = "tidy";
	
	// coalesced thread size, (tidx-(tidx%coalesced_size))
	public final static String COALESCED_IDX = "coalesced_idx";
	public final static String COALESCED_IDY = "coalesced_idy";
	

	String id;
	
	public ThreadIndex(String id) {
		this.id = id;
	}
	
	public static ThreadIndex getThreadIndex(String id) {
		if (IDX.equals(id)) return new ThreadIndex(IDX);
		if (IDY.equals(id)) return new ThreadIndex(IDY);
		if (TIDX.equals(id)) return new ThreadIndex(TIDX);
		if (TIDY.equals(id)) return new ThreadIndex(TIDY);
		if (BIDX.equals(id)) return new ThreadIndex(BIDX);
		if (BIDY.equals(id)) return new ThreadIndex(BIDY);
		if (COALESCED_IDX.equals(id)) return new ThreadIndex(COALESCED_IDX);
		if (COALESCED_IDY.equals(id)) return new ThreadIndex(COALESCED_IDY);
		if (BLOCK_DIM_X.equals(id)) return new ThreadIndex(BLOCK_DIM_X);
		if (BLOCK_DIM_Y.equals(id)) return new ThreadIndex(BLOCK_DIM_Y);
		return null;
	}

	public String getId() {
		return id;
	}
	
	public void setId(String id) {
		this.id = id;
	}

	public String toString() {
		return "ThreadIndex="+getCoefficient()+":"+id;
	}
	
    public boolean equals(Object obj) {
    	if (!(obj instanceof ThreadIndex)) return false;
    	ThreadIndex index = (ThreadIndex)obj;
    	return (this.id.equals(index.id)&&this.getCoefficient()==index.getCoefficient());
	}
	public Object clone() {
		ThreadIndex ti = (ThreadIndex)super.clone();
		ti.setCoefficient(this.getCoefficient());
		return ti;
	}
}
