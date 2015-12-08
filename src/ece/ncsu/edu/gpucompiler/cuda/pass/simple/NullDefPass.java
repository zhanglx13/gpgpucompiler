package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.List;

import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Identifier;
import cetus.hir.VariableDeclaration;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.VariableTools;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class NullDefPass extends Pass {



	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}


	@Override
	public void dopass(GProcedure proc) {

		DepthFirstIterator dfi = new DepthFirstIterator(proc.getProcedure());
		List<DeclarationStatement> dss = dfi.getList(DeclarationStatement.class);
		for (int i=0; i<dss.size(); i++) {
			for (int j=i+1; j<dss.size(); j++) {
				DeclarationStatement dsi = dss.get(i);
				DeclarationStatement dsj = dss.get(j);
				if (!(dsj.getDeclaration() instanceof VariableDeclaration)) continue; 
				if (!(dsi.getDeclaration() instanceof VariableDeclaration)) continue;
				VariableDeclaration vdi = (VariableDeclaration)dsi.getDeclaration();
				VariableDeclaration vdj = (VariableDeclaration)dsj.getDeclaration();
				String iname = VariableTools.getName(vdi).toString();
				String jname = VariableTools.getName(vdj).toString();
				if (iname.equals(jname)&&dsj.getParent()==dsi.getParent()) {
					dsj.detach();
					dss.remove(j);
					j--;
				}
			}
		}

		
		dfi = new DepthFirstIterator(proc.getProcedure());
		dss = dfi.getList(DeclarationStatement.class);
		for (int i=0; i<dss.size(); i++) {		
			DeclarationStatement dsi = dss.get(i);
			if (!(dsi.getDeclaration() instanceof VariableDeclaration)) continue;
			VariableDeclaration vdi = (VariableDeclaration)dsi.getDeclaration();
			Identifier iname = VariableTools.getName(vdi);
			int count = 0;
			DepthFirstIterator iter = new DepthFirstIterator(dsi.getParent());
			List<Identifier> ids = iter.getList(Identifier.class);
			for (Identifier id: ids) {
				if (id.toString().equals(iname.toString())) {
					count++;
				}
			}
			if (count==1) dsi.detach();
		}
	}

}
