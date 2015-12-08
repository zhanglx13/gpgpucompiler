package ece.ncsu.edu.gpucompiler.cuda.pass;


import java.util.logging.Level;
import java.util.logging.Logger;

import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;

public abstract class Pass {

	int id;
	
	Logger logger = Logger.getLogger("Pass");
	
	public abstract String getName();

	public abstract void dopass(GProcedure proc) throws UnsupportedCodeException ;
	
	public void log(Level level, String msg) {
		logger.log(level, "["+getName()+"]"+msg);
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}
	
	
	
//	public void gerenateCopy(String filename, String source) {
//		filename = filename.replaceAll("\\[|\\]|\\.|\\=|\\-|\\*|\\:|\\;", "_");
//		try {
//			File file = File.createTempFile("gcompiler_"+filename+"_", ".cu");
//			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
//			bw.write(source);
//			bw.close();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//
//	}
}
