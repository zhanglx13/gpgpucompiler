package cetus.codegen;

import cetus.hir.*;

public abstract class CodeGenPass
{
  protected Program program;

  protected CodeGenPass(Program program)
  {
    this.program = program;
  }

  public abstract String getPassName();

  public static void run(CodeGenPass pass)
  {
    Tools.printlnStatus(pass.getPassName() + " begin", 1);
    pass.start();
    Tools.printlnStatus(pass.getPassName() + " end", 1);
  }

  public abstract void start();
}
