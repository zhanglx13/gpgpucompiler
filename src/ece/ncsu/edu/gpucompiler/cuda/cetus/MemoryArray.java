package ece.ncsu.edu.gpucompiler.cuda.cetus;

import cetus.hir.Specifier;

public class MemoryArray {
	String name;
	int[] sizes =  new int[3];
	String[] sizeNames = new String[3];
	int dimension;
	Specifier type;

	public final static int MEMORY_NULL = 0;
	public final static int MEMORY_GLOBAL = 1;
	public final static int MEMORY_SHARED = 2;
	int memoryType = MEMORY_NULL;
	
	public MemoryArray() {}
	
	public MemoryArray(String name, int dimension, Specifier type, int memoryType) {
		this.name = name;
		this.dimension = dimension;
		this.type =  type;
		this.memoryType = memoryType;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getDimension() {
		return dimension;
	}
	
	public void setDimension(int dimension) {
		this.dimension = dimension;
	}	
	
	public int getSize(int dim) {
		return sizes[dim];
	}

	public void setSize(int dim, int size) {
		sizes[dim] = size;
	}
	
	public String getName(int dim) {
		return sizeNames[dim];
	}

	public void setSize(int dim, String name) {
		sizeNames[dim] = name;
	}
	
	
	public Specifier getType() {
		return type;
	}

	public void setType(Specifier type) {
		this.type = type;
	}

	public int getMemoryType() {
		return memoryType;
	}

	public void setMemoryType(int memoryType) {
		this.memoryType = memoryType;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		StringBuffer sb = new StringBuffer();
		sb.append("name:"+name+";dimension="+dimension+";type="+type+";");
		return sb.toString();
	}
	
	
}
