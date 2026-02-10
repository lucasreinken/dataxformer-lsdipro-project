import com.opencsv.CSVReader;
import com.opencsv.ICSVWriter;
import com.opencsv.CSVWriter;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.CSVWriterBuilder;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.en.EnglishAnalyzer;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class CsvLuceneTokenizer {

    private static String analyzeText(Analyzer analyzer, String text) throws IOException {
        if (text == null || text.isBlank()) return "";

        TokenStream ts = analyzer.tokenStream("", new StringReader(text));
        CharTermAttribute termAttr = ts.addAttribute(CharTermAttribute.class);

        ts.reset();
        List<String> tokens = new ArrayList<>();
        while (ts.incrementToken()) {
            tokens.add(termAttr.toString());
        }
        ts.end();
        ts.close();

        return String.join(" ", tokens);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("Usage: CsvLuceneTokenizer <folder-with-csv-files>");
            return;
        }

        Path folder = Paths.get(args[0]);

        if (!Files.isDirectory(folder)) {
            throw new IllegalArgumentException("Provided path is not a folder: " + folder);
        }

        Analyzer analyzer = new StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(folder, "*.csv")) {
            for (Path inputPath : stream) {

                String inName = inputPath.getFileName().toString();

                // Skip already tokenized files
                if (inName.endsWith("_tokenized.csv")) {
                    continue;
                }

                String outputName = inName.substring(0, inName.length() - 4) + "_tokenized.csv";
                Path outputPath = inputPath.resolveSibling(outputName);

                System.out.println("Processing: " + inputPath.getFileName());

                try (
                        CSVReader reader = new CSVReaderBuilder(new FileReader(inputPath.toFile()))
                                .withCSVParser(new CSVParserBuilder()
                                        .withSeparator(',')
                                        .build())
                                .build();

                        ICSVWriter writer = new CSVWriterBuilder(new FileWriter(outputPath.toFile()))
                                .withSeparator(',')
                                .build();
                ) {
                    String[] row;
                    while ((row = reader.readNext()) != null) {
                        String[] tokenizedRow = new String[row.length];
                        for (int i = 0; i < row.length; i++) {
                            tokenizedRow[i] = analyzeText(analyzer, row[i]);
                        }
                        writer.writeNext(tokenizedRow);
                    }
                }

                System.out.println(" → Written: " + outputName);
            }
        }

        analyzer.close();
        System.out.println("Done.");
    }
}
