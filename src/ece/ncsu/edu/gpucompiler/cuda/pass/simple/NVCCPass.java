package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.logging.Level;

import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class NVCCPass extends Pass {


	static BufferedWriter logWriter = null;
	static {
		try {
			File log = new File(new File(System.getProperty("java.io.tmpdir")), "nvcc.log");
			logWriter = new BufferedWriter(new FileWriter(log));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public NVCCPass() {}

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}

	@Override
	public void dopass(GProcedure procedure) {
//		if (procedure.getRegCount()>0&&procedure.getSharedMemorySize()>0) return;
		
		try {
			String OS = System.getProperty("os.name").toLowerCase();
			String nvcc = "nvcc.bat";
			if (OS.toLowerCase().indexOf("windows")==-1) {
				nvcc = "./nvcc.sh";
			}
//			System.out.println(OS+"-"+nvcc);
			File file = procedure.gerenateRAWOutput(new File(System.getProperty("java.io.tmpdir")), procedure.getProcedure().getSymbolName()+"_input.cu");
			String cmd = nvcc+" "+file.getAbsolutePath()+"";
			System.out.println(cmd);

			Process proc = Runtime.getRuntime().exec(cmd);
			BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
			BufferedReader error = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
			String line = null;
			int reg = 0;
			int smem = 0;

			while ((line=error.readLine())!=null) {
				if (line.startsWith("ptxas info")) {
					int regStart = line.indexOf("Used");
					if (regStart!=-1) {
						int regEnd = line.indexOf("registers", regStart);
						if (regEnd!=-1) {
							reg = Integer.parseInt(line.substring(regStart+4, regEnd).trim());
							int memStart = line.indexOf(",", regEnd);
							if (memStart!=-1) {
								int memEnd = line.indexOf("bytes", memStart);
								if (memEnd!=-1) {
									String s = line.substring(memStart+1, memEnd).trim();
									int i = s.indexOf('+');
									int m0 = Integer.parseInt(s.substring(0, i).trim());
									int m1 = Integer.parseInt(s.substring(i+1).trim());
									smem = m0+m1;
									
								}
							}					
						}
						procedure.setRegCount(reg);
						procedure.setSharedMemorySize(smem);
						log(Level.INFO, "reg="+reg+";smem="+smem);
						break;
					}
				}
				logWriter.write(line+"\n");
//				System.out.println(line);
			}
			logWriter.flush();
			
			while ((line=reader.readLine())!=null) {
//				System.out.println(line);

			}
			
		}
		catch (IOException ex) {
			ex.printStackTrace();
		}
	}
}
