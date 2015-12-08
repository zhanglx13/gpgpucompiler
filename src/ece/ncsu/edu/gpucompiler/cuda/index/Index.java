package ece.ncsu.edu.gpucompiler.cuda.index;


public class Index implements Cloneable {
	
//	protected boolean negative = false;

	private int coefficient = 1;
	
	
	
	public int getCoefficient() {
		return coefficient;
	}

	public void setCoefficient(int coefficient) {
		this.coefficient = coefficient;
	}

//	public boolean isNegative() {
//		return negative;
//	}
//
//	public void setNegative(boolean negative) {
//		this.negative = negative;
//	}
	

	
	public String toString() {
		return "coe="+getCoefficient();
	}
	
	public Object clone() {
		Index obj = null;
		try {
			obj = (Index)super.clone();
			obj.setCoefficient(this.getCoefficient());
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return obj;
	}
}
