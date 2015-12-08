package ece.ncsu.edu.gpucompiler.cuda.index;

import cetus.hir.Identifier;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;

public class LoopIndex extends Index {
	GLoop loop;
	Identifier id;
	
	public LoopIndex(Identifier id) {
		this.id = id;
	}
	
	public Identifier getId() {
		return id;
	}
	public void setId(Identifier id) {
		this.id = id;
	}
	public GLoop getLoop() {
		return loop;
	}
	public void setLoop(GLoop loop) {
		this.loop = loop;
	}

	public String toString() {
		return super.toString()+";LoopIndex="+id;
	}	
	
	public Object clone() {
		LoopIndex obj = (LoopIndex)super.clone();
		obj.setId((Identifier)id.clone());
		obj.setCoefficient(this.getCoefficient());
		return obj;
	}
	
    public boolean equals(Object obj) {
    	if (!(obj instanceof LoopIndex)) return false;
    	LoopIndex index = (LoopIndex)obj;
    	return (this.id.equals(index.id)&&this.getCoefficient()==index.getCoefficient());
	}	
}
