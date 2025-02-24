package again250224;

import java.util.Arrays;
import java.util.List;

/**
 * 设计有序流
 */
public class OrderedStream {

    private final String[] stream;
    private int ptr = 1;

    public OrderedStream(int n) {
        stream = new String[n + 1];
    }

    public List<String> insert(int idKey, String value) {
        stream[idKey] = value;
        int start = ptr;
        while (ptr < stream.length && stream[ptr] != null) {
            ptr++;
        }
        return Arrays.asList(stream).subList(start, ptr);
    }
}
