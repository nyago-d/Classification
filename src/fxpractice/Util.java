/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fxpractice;

import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import javafx.util.Pair;

/**
 *
 * @author daiki
 */
public class Util {
    
//    public static Stream<Pair<Double, Double>> zip(DoubleStream list1, DoubleStream list2) {
//        list1.
//    }
//    
//    private static <T, S> Stream<Pair<T, S>>
    
    public static <T, S> Stream<Pair<T, S>> zip(Stream<T> list1, Stream<S> list2) {
        
        Iterator<T> fite = list1.iterator();
        Iterator<S> site = list2.iterator();
        
        Iterator<Pair<T, S>> iterator = new Iterator<Pair<T, S>>(){
            @Override
            public boolean hasNext() {
                return fite.hasNext() && site.hasNext();
            }
            @Override
            public Pair<T, S> next() {
                return new Pair(fite.next(), site.next());
            }
        };
        
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(
                        iterator, Spliterator.NONNULL | Spliterator.ORDERED), false);
    }
}
