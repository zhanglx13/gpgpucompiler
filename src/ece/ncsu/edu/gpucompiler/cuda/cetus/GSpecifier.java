package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.Specifier;

/**
 * The class supports multiple specifiers.
 * we support to 3 specifiers: cudaSpecifier additionSpecifier baseSpecifier. For example, __shared__ unsigned int
 * the last one must be baseSpecifier
 * @author jack
 *
 */
public class GSpecifier extends Specifier {

//	List<Specifier> specifiers = new ArrayList<Specifier>();
	Specifier cudaSpecifier = null;
	Specifier baseSpecifier = null;
	Specifier additionSpecifier = null;
	
	
	
	public GSpecifier(Specifier specifier) {
		List<Specifier> specifiers = new ArrayList<Specifier>();
		specifiers.add(specifier);
		init(specifiers);
	}
	
	public GSpecifier(List<Specifier> specifiers) {
		init(specifiers);
	}
	
	void init(List<Specifier> specifiers) {
		cudaSpecifier = initCudaSpecifier(specifiers);	
		additionSpecifier = initAdditionSpecifier(specifiers);
		if (specifiers.size()>0) baseSpecifier = specifiers.get(specifiers.size()-1);
	}
	
	Specifier initCudaSpecifier(List<Specifier> specifiers) {
		if (specifiers.size()>0) {
			Specifier spe = specifiers.get(0);
			if (spe==GLOBAL) return GLOBAL;
			if (spe==LOCAL) return LOCAL;
			if (spe==SHARED) return SHARED;
			if (spe==CONSTANT) return CONSTANT;
			if (spe==DEVICE) return DEVICE;
			if (spe==HOST) return HOST;
		}
		return null;
	}
	
	Specifier initAdditionSpecifier(List<Specifier> specifiers) {
		for (Specifier spe: specifiers) {
			if (spe==UNSIGNED) return UNSIGNED;
			if (spe==SIGNED) return SIGNED;
		}
		return null;		
	}


	public Specifier getCudaSpecifier() {
		return cudaSpecifier;
	}

	public void setCudaSpecifier(Specifier cudaSpecifier) {
		this.cudaSpecifier = cudaSpecifier;
	}

	public Specifier getBaseSpecifier() {
		return baseSpecifier;
	}

	public void setBaseSpecifier(Specifier baseSpecifier) {
		this.baseSpecifier = baseSpecifier;
	}
	

	
	
	
}
