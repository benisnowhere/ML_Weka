package wekaapp;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;
import weka.core.AttributeStats;

/**
 *
 * @author tegar
 */
public class MyJ48
        extends Classifier {

    Instances instances;
    /**
     * for serialization
     */
    static final long serialVersionUID = -2693678647096322561L;

    /**
     * The node's Children.
     */
    private MyJ48[] m_Children;

    /**
     * Attribute used for splitting.
     */
    private Attribute m_Attribute;

    /**
     * Class value if node is leaf.
     */
    private double m_ClassValue;

    /**
     * Class distribution if node is leaf.
     */
    private double[] m_ClassDistribution;

    /**
     * Class attribute of dataset.
     */
    private Attribute m_ClassAttribute;


    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // handling missing value
        Instances noMissingValueData = handleMissingValue(data);

        for (int i = 0; i < noMissingValueData.numInstances(); i++) {
            System.out.println(noMissingValueData.instance(i).toString());
        }
        makeTree(noMissingValueData);
    }

    public int maxAttr(Instances data, Attribute atr) throws Exception {
        int[] maxval = new int[atr.numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance temp = data.instance(i);
            maxval[(int) temp.classValue()]++;
        }
        return Utils.maxIndex(maxval);
    }

    private Instances handleMissingValue(Instances _data) {

        Instances data = _data;
        Enumeration attrEnum = data.enumerateAttributes();
        while (attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            //Handling nominal, just assign it with majority value
            if (attr.isNominal()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                int maxIndex = 0;
                for (int i = 1; i < attr.numValues(); i++) {
                    if (attributeStats.nominalCounts[maxIndex] < attributeStats.nominalCounts[i]) {
                        maxIndex = i;
                    }
                }
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance instance = (Instance) data.instance(i);
//                    System.out.println(instance.toString());
                    if (instance.isMissing(attr.index())) {
                        System.out.println("Yes");
                        instance.setValue(attr.index(), maxIndex);
                    }
                }
            } //Handling numeric, just assign it with mean of attribute's instances
            else if (attr.isNumeric()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                double mean = attributeStats.numericStats.mean;
                if (Double.isNaN(mean)) {
                    mean = 0;
                }
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance instance = (Instance) data.instance(i);
                    if (instance.isMissing(attr.index())) {
                        instance.setValue(attr.index(), (int) mean);
                    }
                }
            }
        }
        return data;
    }

    private Instance handleMissingValue(Instance _data) {

        Instance instance = _data;
        Instances data = instances;
        Enumeration attrEnum = data.enumerateAttributes();
        while (attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            //Handling nominal, just assign it with majority value
            if (attr.isNominal()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                int maxIndex = 0;
                for (int i = 1; i < attr.numValues(); i++) {
                    if (attributeStats.nominalCounts[maxIndex] < attributeStats.nominalCounts[i]) {
                        maxIndex = i;
                    }
                }
                if (instance.isMissing(attr.index())) {
                    System.out.println("Yes");
                    instance.setValue(attr.index(), maxIndex);
                }
            } //Handling numeric, just assign it with mean of attribute's instances
            else if (attr.isNumeric()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                double mean = attributeStats.numericStats.mean;
                if (Double.isNaN(mean)) {
                    mean = 0;
                }
                if (instance.isMissing(attr.index())) {
                    instance.setValue(attr.index(), (int) mean);
                }
            }
        }
        return instance;
    }

    private void makeTree(Instances data) throws Exception {
        // Check if no instances have reached this node.
        instances = data;
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = -1;
            m_ClassDistribution = new double[data.numClasses()];
            return;
        }

        // Compute attribute with maximum gain ratio
        double[] gainRatio = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            if (att.isNominal()) {
                //kasus normal
                gainRatio[att.index()] = computeGainRatio(data, att);
            } else if (att.isNumeric()) {
                //kasus tidak normal
                gainRatio[att.index()] = computeGainRatio(data, att, getOptimumThreshold(data, att));
            }
        }
        // Make leaf if gain ratio is zero. 
        // Otherwise create successors.
        if (Utils.eq(gainRatio[Utils.maxIndex(gainRatio)], 0)) {
            m_Attribute = null;
            m_ClassDistribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                m_ClassDistribution[(int) inst.classValue()]++;
            }
            Utils.normalize(m_ClassDistribution);
            m_ClassValue = Utils.maxIndex(m_ClassDistribution);
            m_ClassAttribute = data.classAttribute();
        } else {
            m_Attribute = data.attribute(Utils.maxIndex(gainRatio));
            Instances[] splitData;
            int numChild;
            if (m_Attribute.isNominal()) {
                numChild = m_Attribute.numValues();
                splitData = splitData(data, m_Attribute);
            } else {
                numChild = 2;
                splitData = splitData(data, m_Attribute, getOptimumThreshold(data, m_Attribute));
            }
            m_Children = new MyJ48[numChild];
            for (int j = 0; j < numChild; j++) {
                m_Children[j] = new MyJ48();
                m_Children[j].makeTree(splitData[j]);
                if (Utils.eq(splitData[j].numInstances(), 0)) {
                    m_Children[j].m_ClassValue = maxAttr(data, data.classAttribute());
                }
            }

            for (int i = 0; i < numChild; i++) {
                if (m_Children[i].m_ClassValue != 0 && Utils.eq(m_Children[i].m_ClassValue, -999)) {
                    double[] classDistribution = new double[data.numClasses()];
                    Enumeration instanceEnum = data.enumerateInstances();
                    while (instanceEnum.hasMoreElements()) {
                        Instance instance = (Instance) instanceEnum.nextElement();
                        classDistribution[(int) instance.classValue()]++;
                    }
                    Utils.normalize(classDistribution);
                    m_Children[i].m_ClassValue = Utils.maxIndex(classDistribution);
                    m_Children[i].m_ClassDistribution = classDistribution;
                }
            }
            pruneTree();
        }
    }

    @Override
    public double classifyInstance(Instance instance)
            throws Exception {

        if (instance.hasMissingValue()) {
            throw new Exception("Unexpected Error");
        }
        if (m_Attribute == null) {
            {
                if (!Utils.eq(m_ClassValue, Double.NaN)) {
                    return m_ClassValue;
                } else {
                    Enumeration a = instance.enumerateAttributes();
                    return instance.value(m_ClassAttribute);
                }
            }
        } else {
            if (m_Attribute.isNumeric()) {
                int numericAttrIdx = -1;
                if (instance.value(m_Attribute) > getOptimumThreshold(instances, m_Attribute)) {
                    numericAttrIdx = 1;
                } else {
                    numericAttrIdx = 0;
                }
                return m_Children[(int) numericAttrIdx].
                        classifyInstance(instance);
            } else if (m_Attribute.isNominal()) {
                return m_Children[(int) instance.value(m_Attribute)].
                        classifyInstance(instance);
            } else {
                throw new Exception("Unexpected Error");
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance)
            throws Exception {
        System.out.println("Instance = " + instance.toString());
        if (m_Attribute != null) {
            double m_Attribute_idx = 0;
            if (m_Attribute.isNominal()) {
                m_Attribute_idx = instance.value(m_Attribute);
                if (Double.isNaN(m_Attribute_idx)) {
                    Instances[] instancesSplitted = splitData(instances, m_Attribute);
                    int largestNumIdx = -1;
                    int cnt = 0;
                    for (int i = 0; i < instancesSplitted.length; ++i) {
                        int tmp = instancesSplitted[i].numInstances();
                        if (tmp > cnt) {
                            largestNumIdx = i;
                        }
                    }
                    m_Attribute_idx = largestNumIdx;
                }
                if (m_Attribute_idx == -1) {
                    throw new Exception("Unexpected Error");
                }
            } else if (m_Attribute.isNumeric()) {
                double val = instance.value(m_Attribute);
                if (Double.isNaN(val)) {
                    instance = handleMissingValue(instance);
                    System.out.println(val);
                    System.out.println(instance.toString());
                    val = instance.value(m_Attribute);
                }
                //manual classifying
                if (val >= getOptimumThreshold(instances, m_Attribute)) {
                    m_Attribute_idx = 1;
                } else {
                    m_Attribute_idx = 0;
                }
            }
            if (m_Children.length > 0) {
                return m_Children[(int) m_Attribute_idx].distributionForInstance(instance);
            }
            if (m_ClassDistribution != null) {
                return m_ClassDistribution;
            } else {
                System.out.println("Halo sayang");
            }
        } else {
            return m_ClassDistribution;
        }
        if (m_ClassDistribution != null) {
            return m_ClassDistribution;
        } else {
            return null;
        }
    }

    @Override
    public String toString() {

        if ((m_ClassDistribution == null) && (m_Children == null)) {
            return "MyJ48: No model built yet.";
        }
        return "MyJ48\n\n" + toString(0);
    }
    
    public double computeGainRatio(Instances data, Attribute attr) throws Exception {

        double infoGain = 0.0;
        Instances[] splitData = MyJ48.this.splitData(data, attr);
        infoGain = computeEntropy(data);
        for (int i = 0; i < attr.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= (double) splitData[i].numInstances()
                        / (double) data.numInstances() * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public double computeGainRatio(Instances data, Attribute attr, double threshold) throws Exception {

        double infoGain = 0.0;
        Instances[] splitData = splitData(data, attr, threshold);
        infoGain = computeEntropy(data);
        for (int i = 0; i < 2; i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain = infoGain - (double) splitData[i].numInstances()
                        / (double) data.numInstances() * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public Instances[] splitData(Instances data, Attribute attr, double threshold) throws Exception {
        Instances[] splitedData = new Instances[2];
        for (int i = 0; i < 2; i++) {
            splitedData[i] = new Instances(data, data.numInstances()); // initialize with data template and max capacity
        }

        Enumeration instanceIterator = data.enumerateInstances();
        while (instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            if (instance.value(attr) >= threshold) {
                splitedData[1].add(instance);
            } else {
                splitedData[0].add(instance);
            }
        }

        for (Instances instances : splitedData) {
            instances.compactify(); //WEKA said it so
        }

        return splitedData;
    }

    public Instances[] splitData(Instances data, Attribute attr) {

        Instances[] splitedData = new Instances[attr.numValues()];
        for (int i = 0; i < attr.numValues(); i++) {
            splitedData[i] = new Instances(data, data.numInstances());
        }

        Enumeration instanceIterator = data.enumerateInstances();
        while (instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            splitedData[(int) instance.value(attr)].add(instance);
        }

        for (Instances instances : splitedData) {
            instances.compactify(); //WEKA said it so, for the sake of optimizing
        }

        return splitedData;
    }

    public double computeEntropy(Instances data) {
        // This fucking validation is a must
        if (data.numInstances() == 0) {
            return 0.0;
        }

        double[] classCounts = new double[data.numClasses()];
        Enumeration instanceIterator = data.enumerateInstances();
        int totalInstance = 0;
        while (instanceIterator.hasMoreElements()) {
            Instance inst = (Instance) instanceIterator.nextElement();
            classCounts[(int) inst.classValue()]++;
            totalInstance++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            double fraction = classCounts[j] / totalInstance;
            if (fraction != 0) {
                entropy -= fraction * Utils.log2(fraction);
            }
        }

        return entropy;
    }

    private double getOptimumThreshold(Instances data, Attribute attribute) throws Exception {
        double[] threshold = new double[data.numInstances()];
        double[] gainRatio = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances() - 1; ++i) {
            if (data.instance(i).classValue() != data.instance(i + 1).classValue()) {
                threshold[i] = (data.instance(i).value(attribute) + data.instance(i + 1).value(attribute)) / 2;
                gainRatio[i] = computeGainRatio(data, attribute, threshold[i]);
            }
        }
        double result = (double) threshold[Utils.maxIndex(gainRatio)];
        return result;
    }

    private String toString(int level) {

        StringBuilder text = new StringBuilder();

        if (m_Attribute == null) {
            if (Instance.isMissingValue(m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": ").append(m_ClassAttribute.value((int) m_ClassValue));
                for (int i=0; i< m_ClassDistribution.length;i++) {
                    text.append(" " +  m_ClassDistribution[i]);
                }
                
            }
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name()).append(" = ").append(m_Attribute.value(j));
                text.append(m_Children[j].toString(level + 1));
            }
        }
        return text.toString();
    }
    
    public double computeError(Instances instances) throws Exception {
        int correctInstances = 0;
        int incorrectInstances = 0;
        Enumeration enumeration = instances.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            Instance instance = (Instance) enumeration.nextElement();
            if (instance.classValue() == classifyInstance(instance)) {
                correctInstances++;
            } else {
                incorrectInstances++;
            }
        }
        return (double) incorrectInstances / (double) (incorrectInstances + correctInstances);
    }

    private void pruneTree() throws Exception {
        //Prepruning, prune before its too late, beybeh
        if (m_Children != null) {
            double beforePruningError = this.computeError(instances);

            double[] classDistribution = new double[instances.numClasses()];
            Enumeration instanceEnum = instances.enumerateInstances();
            while (instanceEnum.hasMoreElements()) {
                Instance instance = (Instance) instanceEnum.nextElement();
                classDistribution[(int) instance.classValue()]++;
            }
            Utils.normalize(classDistribution);
            int idxClass = Utils.maxIndex(classDistribution);

            int correctInstances = 0;
            int incorrectInstances = 0;
            Enumeration enumeration = instances.enumerateInstances();
            while (enumeration.hasMoreElements()) {
                Instance instance = (Instance) enumeration.nextElement();
                if (instance.classValue() == classifyInstance(instance)) {
                    correctInstances++;
                } else {
                    incorrectInstances++;
                }
            }
            double afterPruningError = (double) incorrectInstances / (double) (correctInstances + incorrectInstances);
            if (beforePruningError > afterPruningError) {
                System.out.println("Pruning, behold the power");
                m_Children = null;
                m_Attribute = null;
                m_ClassValue = idxClass;
                m_ClassDistribution = classDistribution;
            }

        }

    }

}