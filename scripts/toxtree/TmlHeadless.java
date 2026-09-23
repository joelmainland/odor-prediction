import java.beans.XMLDecoder;
import java.io.*;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.io.IChemObjectWriter;
import org.openscience.cdk.io.iterator.IIteratingChemObjectReader;
import toxTree.core.IDecisionMethod;
import toxTree.core.IDecisionResult;
import toxTree.query.MolAnalyser;

/** Same pipeline as toxTree.apps.ToxtreeHeadless, but the decision tree
 *  comes from a user-defined .tml (XMLDecoder) file instead of a class name.
 *  usage: TmlHeadless tree.tml in.csv out.csv */
public class TmlHeadless {
  public static void main(String[] a) throws Exception {
    XMLDecoder d = new XMLDecoder(new BufferedInputStream(new FileInputStream(a[0])));
    IDecisionMethod method = (IDecisionMethod) d.readObject();
    d.close();
    IDecisionResult result = method.createDecisionResult();
    IIteratingChemObjectReader reader = ambit2.core.io.FileInputState.getReader(new File(a[1]));
    File out = new File(a[2]);
    IChemObjectWriter writer = ambit2.core.io.FileOutputState.getWriter(new FileOutputStream(out), out.getName());
    int n = 0, err = 0;
    while (reader.hasNext()) {
      IAtomContainer mol = (IAtomContainer) reader.next();
      try {
        MolAnalyser.analyse(mol);
        result.classify(mol);
        result.assignResult(mol);
      } catch (Exception e) { err++; }
      writer.write(mol);
      n++;
    }
    reader.close(); writer.close();
    System.err.println(method.getTitle() + ": processed " + n + ", errors " + err);
  }
}
