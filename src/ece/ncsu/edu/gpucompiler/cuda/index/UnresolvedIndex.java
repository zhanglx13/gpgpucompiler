package ece.ncsu.edu.gpucompiler.cuda.index;

import cetus.hir.Identifier;

public class UnresolvedIndex extends Index {

	Identifier id;
	
	public UnresolvedIndex(Identifier id) {
		this.id = id;
	}

	public Identifier getId() {
		return id;
	}

	public void setId(Identifier id) {
		this.id = id;
	}
	
	public Object clone() {
		UnresolvedIndex obj = (UnresolvedIndex)super.clone();
		obj.setId((Identifier)id.clone());
		obj.setCoefficient(this.getCoefficient());
		return obj;
	}
	
	public String toString() {
		return "UnresolvedIndex="+id;
	}
}
