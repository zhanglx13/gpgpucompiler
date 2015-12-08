package ece.ncsu.edu.gpucompiler.cuda;

import java.io.File;
import java.util.List;

import cetus.exec.Driver;
import cetus.hir.Annotation;
import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Procedure;
import cetus.hir.Statement;
import cetus.hir.VariableDeclaration;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CudaConfig;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;
import ece.ncsu.edu.gpucompiler.cuda.pass.CoalescedPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.IteratorPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.MergePass;
import ece.ncsu.edu.gpucompiler.cuda.pass.PartitionPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.PrefetchingPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.RAWPass;
import ece.ncsu.edu.gpucompiler.cuda.pass.VectorizationPass;

public class KernelDriver extends Driver {
	static {
		options.add("partition", "--apply partition pass");
		options.add("iterator", "--apply iterator pass");
		options.add("vectorization", "--apply vectorization pass");
		options.add("merge0", "--apply merge pass, default opened");
		options.add("merge1", "--apply additional merge pass");
		options.add("prefetching", "--apply prefetching pass");
		options.add("raw", "--apply read after write pass");
		options.add("temp", "--temp folder");
		options.add("output", "--output folder");
		options.add("cuda", "--CUDA version");
		 
//		System.setProperty("java.io.tmpdir", new java.io.File("./temp").getAbsolutePath());
		
		
	}
	
	
	public static void main(String[] args)
	{
		try {
			KernelDriver dr = (new KernelDriver());
//			dr.run(args);
			dr.doOptimize(args);
		}
		catch (Exception ex) {
			System.out.println(ex.getMessage());
			ex.printStackTrace();
		}
	}
	

	public void doOptimize(String[] args) throws UnsupportedCodeException {
		super.parseCommandLine(args);
		
		if (options.getValue("cuda")!=null) {
			CudaConfig.setDefault(CudaConfig.get(options.getValue("cuda")));
		}

		File temp = null;
		if (options.getValue("temp")!=null) {
			temp = new java.io.File(options.getValue("temp"));
		}
		else {
			temp = new java.io.File("temp");
		}
		if (!temp.exists()) temp.mkdir();
		System.setProperty("java.io.tmpdir", temp.getAbsolutePath());
		
		File output = null;
		if (options.getValue("output")!=null) {
			output = new java.io.File(options.getValue("output"));
		}
		else {
			output = new java.io.File("output");
		}
		
		if (!output.exists()) output.mkdir();
		String file = super.filenames[0];
		GProcedure proc =  new GProcedure(file);
		proc.setOutput(output);
		int id = 0;
		{
			if (options.getValue("vectorization")!=null) {
				System.out.println("start vectorization");
				VectorizationPass vpass = new VectorizationPass();
				vpass.setId(id++);
				vpass.dopass(proc);
				proc.refresh();
			}
			
			System.out.println("start coalescing");
			CoalescedPass cpass = new CoalescedPass();
			cpass.setId(id++);
			cpass.dopass(proc);
			proc.refresh();
//			System.out.println(proc.getProcedure().toString());
			
			System.out.println("start MergePass");
			MergePass mpass = new MergePass();
			if (options.getValue("merge0")!=null&&options.getValue("merge0").length()!=0) {
				String[] xy = options.getValue("merge0").split(":");
				mpass.setxNumber(Integer.parseInt(xy[0]));
				mpass.setyNumber(Integer.parseInt(xy[1]));
				System.out.println("specify merge0 number : "+mpass.getxNumber()+","+mpass.getyNumber());
				
			}
			mpass.setId(id++);
			mpass.dopass(proc);
			proc.refresh();
			
			if (options.getValue("merge1")!=null) {
				if (options.getValue("merge1").length()!=0) {
					String[] xy = options.getValue("merge1").split(":");
					mpass.setxNumber(Integer.parseInt(xy[0]));
					mpass.setyNumber(Integer.parseInt(xy[1]));
					System.out.println("specify merge1 number : "+mpass.getxNumber()+","+mpass.getyNumber());
				}
				mpass.setId(id++);
				mpass.dopass(proc);
				proc.refresh();
			}
			if (options.getValue("iterator")!=null) {
				IteratorPass itpass = new IteratorPass();
				itpass.setId(id++);
				itpass.dopass(proc);
				proc.refresh();
			}
			
			if (options.getValue("raw")!=null) {
				RAWPass rawpass = new RAWPass();
				rawpass.setId(id++);
				rawpass.dopass(proc);
			}
			
			
			if (options.getValue("partition")!=null) {
				PartitionPass ppass = new PartitionPass();
				ppass.setId(id++);
				ppass.dopass(proc);
			}
			if (options.getValue("prefetching")!=null) {
				PrefetchingPass ppass = new PrefetchingPass();
				ppass.dopass(proc);
			}
			
			
			proc.gerenateOutput(proc.getProcedure().getName().toString()+"_output");
//			System.out.println("start MergePass");
//			mpass.dopass(proc);

//			PrefetchingPass ppass = new PrefetchingPass();
//			ppass.dopass(proc);

//			mpass = new MergePass();
//			mpass.dopass(proc);
		
		}

	}
	
	
	public void analysis(Procedure pro) {
		{
			BreadthFirstIterator bfi = new BreadthFirstIterator(pro);
			List<Annotation> comments = (List<Annotation>)bfi.getList(Annotation.class);
//			Hashtable<String, Long> list = new Hashtable();
			for (Annotation comment: comments) {
				String s = comment.getText();
				System.out.println(s);
			}		
		}
		List sbs = pro.getParameters();
		for (int i=0; i<sbs.size(); i++) {
			VariableDeclaration id = (VariableDeclaration)sbs.get(i);
			System.out.println(id.getSpecifiers().get(0));
			System.out.println(id.getDeclarator(0));
		}
		CompoundStatement cs = pro.getBody();
//		System.out.println(cs.toString());
		List list = cs.getChildren();
		System.out.println(cs.countStatements());
		for (int i=0; i<list.size(); i++) {
			System.out.print("stmt"+i+": ");			
			Statement stmt = (Statement)list.get(i);
			System.out.println(stmt.toString());
			if (stmt instanceof ForLoop) {
				ForLoop loop = (ForLoop)stmt;
				System.out.println(loop.getInitialStatement().toString());
				System.out.println(loop.getStep().toString());
				System.out.println(loop.getCondition().toString());
				Statement stmt1 = loop.getBody();
				AssignmentExpression es = (AssignmentExpression)stmt1.getChildren().get(0).getChildren().get(0);
				Expression ex = es.getLHS();
				if (ex instanceof ArrayAccess) {
					ArrayAccess aa  = (ArrayAccess)ex;
					System.out.println(aa.getArrayName());
					System.out.println(aa.getIndex(0));
				}
				System.out.println(es.getLHS());
				System.out.println(es.getRHS());
			}
			else
			if (stmt instanceof ExpressionStatement){
				AssignmentExpression es = (AssignmentExpression)(((ExpressionStatement)stmt).getExpression());
				Expression ex = es.getLHS();
				if (ex instanceof ArrayAccess) {
					ArrayAccess aa  = (ArrayAccess)ex;
					System.out.println(aa.getArrayName());
					System.out.println(aa.getIndex(0));
				}
				System.out.println(es.getLHS()+es.getLHS().getClass().toString());
				System.out.println(es.getRHS());
				
			}
		}
	}
}
