library(ggplot2)
library(broom)
library(ggrepel)
library(dplyr)
library(ggrepel)  
library(RColorBrewer)  
d_acl <- read_csv("data/raw/delta_15lang.csv") 
model = lm(delta_auc ~ l_distance + cultural_distance , data = d_acl)
summary(model)
data <- d_acl
 
 # Sample 10 random points from the dataset
 set.seed(123) 
 sampled_data <- d_acl %>% sample_n(15)
 
 # Define color palette
 point_color <- 'darkred'          
 highlight_color <- 'deepskyblue4' 
 line_color <- 'dodgerblue2'     
 ci_color <- 'lightblue'           
 label_color <- 'darkgreen'        
 segment_color <- 'grey50'         
 
 
 # Create the scatter plot
 p <- ggplot(d_acl, aes(x = cultural_distance, y = delta_auc)) + 
   geom_point(color = point_color) + 
   geom_smooth(method = "lm", se = TRUE, color = line_color, fill = ci_color) + 
   geom_point(data = sampled_data, aes(x = cultural_distance, y = delta_auc), color = highlight_color) + 
   geom_label_repel( 
     data = sampled_data, 
     aes(x = cultural_distance, y = delta_auc, label = `Language Pair`), 
     nudge_x = 0.03, 
     color = label_color,
     size = 2.3,  
     box.padding = 0.35,  
     point.padding = 0.3,  
     segment.color = segment_color  
   ) +
   labs(
     #title = "Scatterplot of AUC vs Cultural Distance",
     x = "WEIRDness Distance", 
     y = expression(Delta(L[a], L[t]))
   ) +
   theme_minimal() +  
   theme(
     panel.border = element_blank(),      
     panel.grid.major = element_blank(),   
     panel.grid.minor = element_blank(),    
     panel.background = element_blank(),    
     axis.line = element_line(color = "black"),  
     axis.title.x = element_text(size = 8),  
     axis.title.y = element_text(size = 9)  
   ) +
   scale_x_continuous(limits = c(0, 0.15))
 
 print(p)
 
 
 ggsave(filename = "ACL.png", plot = p, dpi = 300, width = 6,  height = 3)