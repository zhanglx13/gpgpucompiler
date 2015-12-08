package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.Hashtable;


public class Snapshot {

	String id;
	Hashtable<String, String> properties = new Hashtable<String, String>();
	String procedure;
	String type;
	
	public Snapshot() {}
	
	public Snapshot(String id, String procedure) {
		this.id = id;
		this.procedure = procedure;
	}
	
	public void setProperty(String key, String value) {
		properties.put(key, value);
	}

	public String getProperty(String key) {
		return properties.get(key);
	}
	
	
	
	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public String getId() {
		return id;
	}
	public void setId(String id) {
		this.id = id;
	}
	public String getProcedure() {
		return procedure;
	}
	public void setProcedure(String procedure) {
		this.procedure = procedure;
	}
	
	
	
}
