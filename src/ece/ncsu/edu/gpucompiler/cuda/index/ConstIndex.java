package ece.ncsu.edu.gpucompiler.cuda.index;

public class ConstIndex extends Index {
//	long value = 0;
	
	public ConstIndex() {}
	
	public ConstIndex(int value) {
		setCoefficient(value);
	}
//
//	public long getValue() {
//		return value;
//	}
//
//	public void setValue(long value) {
////		if (value<0) {
////			value = -value;
////			negative = true;
////		}
//		this.value = value;
//	}
//	
//	public long getSignedValue() {
//		return isNegative()?-value:value;
//	}
	
	public String toString() {
		return "ConstIndex="+getCoefficient();
	}
	
	public Object clone() {
		ConstIndex ti = (ConstIndex)super.clone();
		ti.setCoefficient(this.getCoefficient());
		return ti;

	}
	
    public boolean equals(Object obj) {
    	if (!(obj instanceof ConstIndex)) return false;
    	ConstIndex index = (ConstIndex)obj;
    	return (this.getCoefficient()==index.getCoefficient());
	}	
}
