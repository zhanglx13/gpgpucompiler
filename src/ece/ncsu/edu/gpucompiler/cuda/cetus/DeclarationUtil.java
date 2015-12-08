package ece.ncsu.edu.gpucompiler.cuda.cetus;

import cetus.hir.DeclarationStatement;
import cetus.hir.Identifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

public class DeclarationUtil {

	public static String getVariableName(VariableDeclaration vd) {
		if (vd.getNumDeclarators()>1) {
			throw new RuntimeException("We don't support multi declarator now: "+vd.toString());
		}
		VariableDeclarator declarator = (VariableDeclarator)vd.getDeclarator(0);
		return ((Identifier)declarator.getDirectDeclarator()).getName();		
		
	}
	
	public static DeclarationStatement getStatement(VariableDeclaration vd) {
		return ((DeclarationStatement)vd.getParent());
	}
	
	public static GSpecifier getGSpecifier(VariableDeclaration vd) {
		return new GSpecifier(vd.getSpecifiers());
	}
}
