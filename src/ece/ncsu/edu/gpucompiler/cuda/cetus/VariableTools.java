package ece.ncsu.edu.gpucompiler.cuda.cetus;

import cetus.hir.Identifier;
import cetus.hir.SymbolTable;
import cetus.hir.Tools;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

public class VariableTools {

	public static Identifier getName(VariableDeclaration decl) {
		VariableDeclarator declarator = (VariableDeclarator)decl.getDeclarator(0);
		return ((Identifier)declarator.getDirectDeclarator());
	}

	public static Identifier getUnusedSymbol(String prefix, SymbolTable table) {
		String name = null;
		Identifier ident = null;
	
		int i = 0;
		do {
			name = prefix + "_" + i;
			ident = new Identifier(name);
			i++;
		} while (Tools.findSymbol(table, ident) != null);
	
		return ident;
	}
	

}
