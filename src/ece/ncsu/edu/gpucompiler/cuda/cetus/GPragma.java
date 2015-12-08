package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.util.List;

import cetus.hir.Annotation;

public class GPragma {
	public final static String PREFIX = "gCompiler"; 

	public final static String TYPE_BLOCK = "gBlock"; 
	public final static String TYPE_VALUE = "gValue"; 
	public final static String TYPE_MEMORY = "gMemory"; 
	
	Annotation annotation;
	String name;
	List<String> values;
	String type;
	
	
	
	public String getType() {
		return type;
	}
	public void setType(String type) {
		this.type = type;
	}
	public Annotation getAnnotation() {
		return annotation;
	}
	public void setAnnotation(Annotation annotation) {
		this.annotation = annotation;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	protected List<String> getValues() {
		return values;
	}
	protected void setValues(List<String> values) {
		this.values = values;
		update();
	}
	
	Long longvalue = null;
	public long getLongValue() {
		if (longvalue==null) {
			longvalue = new Long(getValues().get(0));
		}
		return longvalue;
	}

	public void setLongValue(long longvalue) {
		this.longvalue = longvalue;
		values.set(0, ""+(longvalue));
		update();
	}

	Integer intvalue = null;
	public int getIntValue() {
		if (intvalue==null) {
			intvalue = new Integer(getValues().get(0));
		}
		return intvalue;
	}

	public void seIntValue(int intvalue) {
		this.intvalue = intvalue;
		values.set(0, ""+(intvalue));
		update();
	}	

	void update() {
		StringBuffer sb = new StringBuffer("#pragma\t"+PREFIX+"\t"+type+"\t"+name);
		for (String v: values) {
			sb.append("\t"+v);
		}
		annotation.setText(sb.toString());
	}
	
}
