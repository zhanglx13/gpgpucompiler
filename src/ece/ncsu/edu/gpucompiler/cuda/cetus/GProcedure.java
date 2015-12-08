package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.StringTokenizer;

import cetus.hir.Annotation;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.FunctionCall;
import cetus.hir.IntegerLiteral;
import cetus.hir.PointerSpecifier;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import ece.ncsu.edu.gpucompiler.cuda.pass.Snapshot;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.Array2FunctionPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.CompoundStatementPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.GLoopPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.VariableDeclarationPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.LoopUnrollPragmaPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.MemoryExpressionPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.NVCCPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.NullDefPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.simple.NullLoopPass;

/**
 * CUDA procedure
 * @author jack
 *
 */
public class GProcedure {
	Hashtable<String, String> defines = new Hashtable();
	Specifier returntype;
	Specifier functype;
	Hashtable<String, MemoryArray> memoryArrays = new Hashtable<String, MemoryArray>();
	Procedure procedure;
	List<GLoop> loops = new ArrayList<GLoop>();
	

	public final static String DEF_blockDimX = "blockDimX";
	public final static String DEF_blockDimY = "blockDimY";
	public final static String DEF_merger_y = "merger_y";
	public final static String DEF_globalDimX = "globalDimX";
	public final static String DEF_globalDimY = "globalDimY";
	public final static String DEF_COALESCED_NUM = "COALESCED_NUM";
	
	static Hashtable<String, String> PRE_DEFINE = new Hashtable();
	static String[] PRE_DEFINE_NAME = {"COALESCED_NUM", "blockDimX", "blockDimY", "gridDimX", "gridDimY", "idx", "idy",
		"bidy", "bidx", "tidx", "tidy", "merger_y", "coalesced_idy", "globalDimX", "globalDimY"};
	
	static {
		PRE_DEFINE.put("COALESCED_NUM", "16");
		PRE_DEFINE.put("blockDimX", "1");
		PRE_DEFINE.put("blockDimY", "1");
		PRE_DEFINE.put("idx", "(blockIdx.x*blockDimX+threadIdx.x)");
		PRE_DEFINE.put("idy", "(blockIdx.y*blockDimY+threadIdx.y)");
		PRE_DEFINE.put("bidy", "(blockIdx.y)");
		PRE_DEFINE.put("bidx", "(blockIdx.x)");
		PRE_DEFINE.put("tidx", "(threadIdx.x)");
		PRE_DEFINE.put("tidy", "(threadIdx.y)");
		PRE_DEFINE.put("gridDimX", "(gridDim.x)");
		PRE_DEFINE.put("gridDimY", "(gridDim.y)");
		PRE_DEFINE.put("merger_y", "1");
		PRE_DEFINE.put("coalesced_idy", "(bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)");
	};
	
	
//	List<Statement> preDefineds = new ArrayList();
//
//	List<String> preDefinedIDs = Arrays.asList("COALESCED_NUM", "idx", "idy", "tidx", "tidy", "coalesced_idy");
		
//	// thread block dimension
//	GPragma blockDimX;
//	GPragma blockDimY;
//
//	// global thread dimension
//	GPragma globalDimX;
//	GPragma globalDimY;
//	
	int regCount;
	int sharedMemorySize;
	int localMemorySize = 0;
	
	
	public void setDef(String name, String value) {
		defines.put(name, value);
	}
	
	public String getDef(String name) {
		return defines.get(name);
	}
	
	public int getDefInt(String name) {
		if (defines.get(name)==null) return Integer.MAX_VALUE;
		return Integer.parseInt(defines.get(name));
	}
//	
//	public List<Statement> getPreDefineds() {
//		return preDefineds;
//	}

	public int getRegCount() {
		if (regCount<=0) return 8;
		return regCount;
	}

	public void setRegCount(int regCount) {
		this.regCount = regCount;
	}

	public int getSharedMemorySize() {
		if (sharedMemorySize<=0) return 16;
		return sharedMemorySize;
	}

	public void setSharedMemorySize(int sharedMemorySize) {
		this.sharedMemorySize = sharedMemorySize;
	}

//	public List<String> getDefines() {
//		return defines;
//	}
//
//	public void setDefines(List<String> defines) {
//		this.defines = defines;
//	}

	List<Snapshot> history = new ArrayList<Snapshot>();

	Hashtable<String, Long> pragmasValue = new Hashtable<String, Long>();
	
	public Hashtable<String, Long> getPragmasValue() {
		return pragmasValue;
	}

	public void setPragmasValue(Hashtable<String, Long> pragmasValue) {
		this.pragmasValue = pragmasValue;
	}



	public List<Snapshot> getHistory() {
		return history;
	}

	public void setHistory(List<Snapshot> history) {
		this.history = history;
	}

	/*
	 * so far read only
	 */
	public int getGlobalDimX() {
		if (defines.get(DEF_globalDimX)==null) return Integer.MAX_VALUE;
		return Integer.parseInt(defines.get(DEF_globalDimX));

	}

	/*
	 * so far read only
	 */
	public int getGlobalDimY() {
		if (defines.get(DEF_globalDimY)==null) return Integer.MAX_VALUE;
		return Integer.parseInt(defines.get(DEF_globalDimY));
	}

	public void setGlobalDimX(int globalDimX) {
		defines.put(DEF_globalDimX, globalDimX+"");
	}

	public void setGlobalDimY(int globalDimY) {
		defines.put(DEF_globalDimY, globalDimY+"");
	}


	
	
	public List<GLoop> getLoops() {
		return loops;
	}

	public void setLoops(List<GLoop> loops) {
		this.loops = loops;
	}
	

	public int getBlockDimY() {
		return Integer.parseInt(defines.get(DEF_blockDimY));
	}

	public int getBlockDimX() {
		return Integer.parseInt(defines.get(DEF_blockDimX));
	}

	public void setBlockDimX(int blockDimX) {
		defines.put(DEF_blockDimX, blockDimX+"");
	}

	public void setBlockDimY(int blockDimY) {
		defines.put(DEF_blockDimY, blockDimY+"");
	}

	String preprocess(String source) {
		StringBuffer sb = new StringBuffer();
		for (String pdef: PRE_DEFINE.keySet()) {
			String s = PRE_DEFINE.get(pdef);
			defines.put(pdef, s);
		}		
		try {
			BufferedReader br = new BufferedReader((new StringReader(source)));
			String line = null;
			while ((line=br.readLine())!=null) {
				line = line.trim();
				if (line.startsWith("#define")) {
//					defines.add(line);
					StringTokenizer st = new StringTokenizer(line);
					if (st.hasMoreElements()) { 
						st.nextElement();
						if (st.hasMoreElements()) {
							String name = (String)st.nextElement();
							if (st.hasMoreElements()) {
								String value =  (String)st.nextElement();
								defines.put(name, value);
							}
						}
					}
				}
				else {
					sb.append(line).append("\n");
				}
			}

			
			int coalescedNumber = getDefInt(DEF_COALESCED_NUM);
			CudaConfig.setCoalescedThread(coalescedNumber);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return sb.toString();
	}
	
	public GProcedure(String file) throws UnsupportedCodeException {
		try {
			BufferedReader br = new BufferedReader((new FileReader(file)));
			String line = null;
			StringBuffer sb = new StringBuffer();
			while ((line=br.readLine())!=null) {
				sb.append(line).append("\n");
			}
			refresh(sb.toString());
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	


	
	public void testResource()throws UnsupportedCodeException  {
		int reqBlocksize = getBlockDimX()*getBlockDimY();
		int reg = CudaConfig.getDefault().getRegisterInMP();
		int smem = CudaConfig.getDefault().getShareMemoryInMP();
		int blocksize = CudaConfig.getDefault().getThreadInBlock();		
		int reqReg = getRegCount();
		int reqSmem = getSharedMemorySize();
		if (reqSmem>smem||reqReg*reqBlocksize>reg||reqBlocksize>blocksize||localMemorySize!=0) {
			throw new UnsupportedCodeException("shared memory, register or block size too larger");
		}
	}
	
	/**
	 * replace the nodes with our nodes to easily access the node information
	 * @throws UnsupportedCodeException 
	 */
	void init() throws UnsupportedCodeException {
//		DepthFirstIterator dfi = new DepthFirstIterator(procedure);
//		List<DeclarationStatement> decs = dfi.getList(DeclarationStatement.class);
//		for (DeclarationStatement ds: decs) {
//			Declaration decl = ds.getDeclaration();
//			if (!(decl instanceof VariableDeclaration)) continue;
//			VariableDeclaration vard = (VariableDeclaration)decl;
//			VariableDeclarator v = (VariableDeclarator)vard.getDeclarator(0);
//			String name = v.getDirectDeclarator().toString();
//			if (this.preDefinedIDs.contains(name)) {
//				this.preDefineds.add(ds);
//				ds.detach();
//			}
//		}		
//		
		new CompoundStatementPass().dopass(this);
		new NullDefPass().dopass(this);
		new NVCCPass().dopass(this);

		this.loops.clear();
		this.pragmasValue.clear();
		this.memoryArrays.clear();

		
//		preDefines;
		
		
		List<Specifier> rts = procedure.getReturnType();
		if (rts.size()>=1) {
			returntype = rts.get(0);
			if (returntype!=Specifier.GLOBAL) {
				throw new UnsupportedCodeException("not cuda kernel function");

			}
		}
		if (rts.size()>=2) {
			functype = rts.get(1);
		}
		
		loadPragma();
		
		memoryArrays = loadMemoryArray(procedure);
		
		
		new VariableDeclarationPass().dopass(this);
		
		
		DepthFirstIterator dfi = new DepthFirstIterator(procedure);
		List<DeclarationStatement> decs = dfi.getList(DeclarationStatement.class);
		for (DeclarationStatement ds: decs) {
			Declaration decl = ds.getDeclaration();
			if (!(decl instanceof VariableDeclaration)) continue;
			VariableDeclaration vard = (VariableDeclaration)decl;
			VariableDeclarator v = (VariableDeclarator)vard.getDeclarator(0);
			String name = v.getDirectDeclarator().toString();
			if (vard.getSpecifiers().size()>0&&vard.getSpecifiers().get(0).equals(Specifier.SHARED)) {
				MemoryArray ma = new MemoryArray();
				ma.setMemoryType(MemoryArray.MEMORY_SHARED);
				ma.setType(vard.getSpecifiers().get(1));
				ma.setName(name);
				ArraySpecifier af = (ArraySpecifier)v.getArraySpecifiers().get(0);
				ma.setDimension(af.getNumDimensions());
				for (int i=0; i<af.getNumDimensions(); i++) {
					IntegerLiteral il = (IntegerLiteral)af.getDimension(i);
					ma.setSize(i, (int)il.getValue());
				}
				memoryArrays.put(ma.getName(), ma);
//				System.out.println("shared:"+ma);
			}

		}
		
		

		
		new GLoopPass().dopass(this);
		
		new NullLoopPass(false).dopass(this);
		new NullLoopPass(true).dopass(this);

		// convert memory expression to our version
		new MemoryExpressionPass().dopass(this);
		
		dfi = new DepthFirstIterator(procedure);
		loops = (List<GLoop>)dfi.getList(GLoop.class);
		
		
//		System.out.println(procedure);
		
		
	}

	@SuppressWarnings("unchecked")
	private static Hashtable<String, MemoryArray> loadMemoryArray(Procedure func) {
		Hashtable<String, MemoryArray> results = new Hashtable<String, MemoryArray>();
		List<VariableDeclaration> sbs = func.getParameters();
		for (int i=0; i<sbs.size(); i++) {
			VariableDeclaration id = sbs.get(i);
			Specifier spec = id.getSpecifiers().get(0);
			Declarator dec = id.getDeclarator(0);
			for (Specifier spe: dec.getSpecifiers()) {
				if (spe instanceof PointerSpecifier) {
					MemoryArray stream = new MemoryArray();
					stream.setName(dec.getSymbol().toString());
					stream.setType(spec);
					stream.setMemoryType(MemoryArray.MEMORY_GLOBAL);
					results.put(dec.getSymbol().toString(), stream);
				}
			}
		}
		
		//map defined function to array
		BreadthFirstIterator bfi = new BreadthFirstIterator(func);
		List<FunctionCall> calls = (List<FunctionCall>)bfi.getList(FunctionCall.class);
		for (FunctionCall fc: calls) {
			String name = fc.getName().toString();
			MemoryArray memory = results.get(name);
			if (memory==null) continue;
			memory.setDimension(fc.getArguments().size());
			List indices = new ArrayList();
			for (Object index: fc.getArguments()) {
				Expression ex = (Expression)index;
				indices.add(ex.clone());
			}
			ArrayAccess aa = new ArrayAccess((Expression)fc.getName().clone(), indices);
			fc.swapWith(aa);
		}
		
		
		return results;
	}

	
	@SuppressWarnings("unchecked")
	Hashtable<String, List<GPragma>> loadPragmaValues() {
		BreadthFirstIterator bfi = new BreadthFirstIterator(procedure);
		List<Annotation> comments = (List<Annotation>)bfi.getList(Annotation.class);
		Hashtable<String, List<GPragma>> list = new Hashtable<String, List<GPragma>>();
		for (Annotation comment: comments) {
			String s = comment.getText();
			StringTokenizer st = new StringTokenizer(s);
			if (st.hasMoreElements()&&"#pragma".equals(st.nextElement())) {
				if (st.hasMoreElements()&&GPragma.PREFIX.equals(st.nextElement())) {
					String type = null, name = null;
					if (st.hasMoreElements()) {
						type = (String)st.nextElement();
						List<GPragma> gps = list.get(type);
						if (gps==null) {
							gps = new ArrayList<GPragma>();
							list.put(type, gps);
						}

//						System.out.println("find:"+type+";"+s);
						List<String> values = new ArrayList<String>();
						if (st.hasMoreElements()) {
							GPragma pragma =  new GPragma();
							name = (String)st.nextElement();
							pragma.setName(name);
							pragma.setType(type);
							pragma.setAnnotation(comment);
							while (st.hasMoreElements()) {
								values.add((String)st.nextElement());
							}
							pragma.setValues(values);
							gps.add(pragma);
						}
					}
				}
			}		
		}
		return list;
	}			
	
	void loadPragma() {
		
		Hashtable<String, List<GPragma>> hash = loadPragmaValues();

		
//		// load thread block dimension
//		List<GPragma> blockPragmas = hash.get(GPragma.TYPE_BLOCK);
//		if (blockPragmas!=null) {
//			for (GPragma gp: blockPragmas) {
//				if (gp.getName().equals("blockDimX")) blockDimX = gp;
//				if (gp.getName().equals("blockDimY")) blockDimY = gp;
//				if (gp.getName().equals("globalDimX")) globalDimX = gp;
//				if (gp.getName().equals("globalDimY")) globalDimY = gp;
//			}
//		}
		
		// load constant value
		List<GPragma> valuePragmas = hash.get(GPragma.TYPE_VALUE);
		if (valuePragmas!=null) {
			for (GPragma gp: valuePragmas) {	// readonly
				pragmasValue.put(gp.getName(), gp.getLongValue());
			}	
		}
		
		
		/*
		// load (global) memory configuration
		List<GPragma> memoryPragmas = hash.get(GPragma.TYPE_MEMORY);
		for (GPragma gp: memoryPragmas) {
			MemoryArray stream = new MemoryArray();
			stream.setName(gp.getName());
			String spe = gp.getValues().get(0);	// specifier
			if (Specifier.FLOAT.toString().equals(spe)) {
				stream.setType(Specifier.FLOAT);
			}
			else
			if (Specifier.INT.toString().equals(spe)) {
				stream.setType(Specifier.INT);
			}
			else {
				stream.setType(new UserSpecifier(new Identifier(spe)));
			}
			stream.setDimension(Integer.parseInt(gp.getValues().get(1)));	// dimension
			memoryArrays.put(stream.getName(), stream);			

		}
		*/
	}
	


	public Specifier getReturntype() {
		return returntype;
	}



	public Specifier getFunctype() {
		return functype;
	}


	public void addMemoryArray(MemoryArray ma) {
		memoryArrays.put(ma.getName(), ma);
	}

	public MemoryArray getMemoryArray(String name) {
		return memoryArrays.get(name);
	}



	public Procedure getProcedure() {
		return procedure;
	}

	public static List<Procedure> getCudaProcedures(Program program) {
		BreadthFirstIterator bfi = new BreadthFirstIterator(program);
		List<Procedure> funcs = bfi.getList(Procedure.class);
		List<Procedure> result = new ArrayList<Procedure>();
		for (Procedure func: funcs) {
//			List sbs = func.getParameters();
			List<Specifier> types = func.getReturnType();
			for (Specifier type: types) {
//				System.out.println(type);
				if (type.equals(Specifier.GLOBAL)) {
					result.add(func);
					continue;
				}
			}
		}
		return result;
	}
	
	public String toString() {
		String s = "functype="+functype+
		";returntype="+returntype+
		";name="+procedure.getName();
		return s;
	}
	
	public void refresh(String source) throws UnsupportedCodeException {
//		System.out.println(source);
		try {
			source = preprocess(source);
			File file = new File(new File(System.getProperty("java.io.tmpdir")), "gcompiler_tmp.cu");
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
//			bw.write("struct float2 { float x; float y;};\n");
			bw.write(source);
			bw.write("\n");
			bw.close();
			List<String> lfile = new ArrayList<String>();
			lfile.add(file.getAbsolutePath());
			Program program = new Program(lfile);
			program.parse();
			this.procedure = getCudaProcedures(program).get(0);
			init();
			
		}
		catch (IOException e) {
			System.err.println("I/O error parsing files");
			System.err.println(e);
			System.exit(1);
		}
		
	}
	
	public void compile() {
		new NVCCPass().dopass(this);
	}
	
	public void refresh() throws UnsupportedCodeException {
		refresh(generateRunnableCode());
	}
	
	public String generateRunnableCode() {
		new Array2FunctionPass(true).dopass(this);
		String source  = procedure.toString();
		new Array2FunctionPass(false).dopass(this);

		StringBuffer sb = new StringBuffer();
		for (String def: PRE_DEFINE_NAME) {
			if (defines.get(def)!=null)
				sb.append("#define "+def+" "+defines.get(def)+"\n");
		}
		for (String def: defines.keySet()) {
			boolean isdone = false;
			for (String def1: PRE_DEFINE_NAME) {
				if (def1.equals(def)) {
					isdone = true;
				}
			}
			if (!isdone)
				sb.append("#define "+def+" "+defines.get(def)+"\n");
		}
		sb.append(source);
		sb.append("\n");
		return sb.toString();
	}
	
	File output;
	
	
	public File getOutput() {
		return output;
	}

	public void setOutput(File output) {
		this.output = output;
	}
	public File gerenateRAWOutput(File folder, String filename) {
		filename = filename.replaceAll("\\[|\\]|\\.|\\=|\\-|\\*|\\:|\\;|\\+|\\(|\\)", "_");
		try {
			File file = new File(folder, "gcompiler_"+filename+".cu");
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			bw.write(generateRunnableCode());
			bw.close();
			return file;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public File gerenateOutput(File folder, String filename) {
		filename = filename.replaceAll("\\[|\\]|\\.|\\=|\\-|\\*|\\:|\\;|\\+|\\(|\\)", "_");
		try {
			File file = new File(folder, "gcompiler_"+filename+".cu");
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			NullDefPass defp =  new NullDefPass();
			defp.dopass(this);
			CompoundStatementPass csp = new CompoundStatementPass();
			csp.dopass(this);
			NullLoopPass nullp = new NullLoopPass(false);
			nullp.dopass(this);
			LoopUnrollPragmaPass lupass = new LoopUnrollPragmaPass();
			lupass.dopass(this);
			bw.write(generateRunnableCode());
			bw.close();
			return file;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public File gerenateOutput(String filename) {
		return gerenateOutput(output, filename);
	}
}
