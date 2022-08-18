package daily;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 设计有序流
 */
public class OrderedStream {

    String[] ss = new String[1010];
    int idx, n;

    public OrderedStream(int _n) {
        Arrays.fill(ss, "");
        idx = 1;n = _n;
    }
    public List<String> insert(int key,String value){
        ss[key] = value;
        List<String> ans = new ArrayList<>();
        while (ss[idx].length() == 5){
            ans.add(ss[idx++]);
        }
        return ans;
    }
}
